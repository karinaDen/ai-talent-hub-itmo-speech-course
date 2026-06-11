import heapq
import math
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


try:
    import soundfile as sf

    def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, buffer_size=4096, backend=None):
        _ = normalize, format, buffer_size, backend
        data, sr = sf.read(str(path), dtype='float32', always_2d=True)
        if frame_offset:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        import torch as _torch
        t = _torch.from_numpy(data.T.copy() if channels_first else data.copy())
        return t, sr

    torchaudio.load = _sf_load
except ImportError:
    pass  


# ---------------------------------------------------------------------------
# Provided utility
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
    
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.word_delimiter_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.word_delimiter
        )
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _lm_score_sentence(self, token_ids: List[int]) -> float:
        """Return KenLM log-prob (natural log) for the full sentence."""
        text = self._ids_to_text(token_ids)
        if not text:
            return 0.0
        return self.lm_model.score(text, bos=True, eos=True) * math.log(10)

    def _lm_score_word(self, word: str, state_in: kenlm.State) -> Tuple[float, kenlm.State]:
        """Score a single word incrementally; return (ln_prob, next_state)."""
        state_out = kenlm.State()
        log10_prob = self.lm_model.BaseScore(state_in, word.lower(), state_out)
        return log10_prob * math.log(10), state_out

    # -----------------------------------------------------------------------
    # Task 1: Greedy decoding
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        token_ids = log_probs.argmax(dim=-1).tolist()  # (T,)

        result, prev = [], None
        for t in token_ids:
            if t != self.blank_token_id and t != prev:
                result.append(t)
            prev = t

        return self._ids_to_text(result)

    # -----------------------------------------------------------------------
    # Task 2: Beam search without LM (CTC prefix beam search)
    # -----------------------------------------------------------------------

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        T, V = log_probs.shape
        blank = self.blank_token_id
        NEG_INF = float('-inf')

    
        beam = {(): (0.0, NEG_INF)}

        for t in range(T):
            lp = log_probs[t]           
            new_beam: dict = {}

            def _update(prefix, p_b, p_nb):
                if prefix in new_beam:
                    ob, onb = new_beam[prefix]
                    new_beam[prefix] = (_log_add(ob, p_b), _log_add(onb, p_nb))
                else:
                    new_beam[prefix] = (p_b, p_nb)

            for prefix, (p_b, p_nb) in beam.items():
                p_total = _log_add(p_b, p_nb)

                _update(prefix, p_total + lp[blank].item(), NEG_INF)

                for c in range(V):
                    if c == blank:
                        continue
                    lpc = lp[c].item()
                    if prefix and prefix[-1] == c:
                        _update(prefix, NEG_INF, p_nb + lpc)
                        _update(prefix + (c,), NEG_INF, p_b + lpc)
                    else:
                        _update(prefix + (c,), NEG_INF, p_total + lpc)

            # Prune to top beam_width hypotheses
            beam = dict(
                sorted(new_beam.items(),
                       key=lambda x: _log_add(x[1][0], x[1][1]),
                       reverse=True)[:self.beam_width]
            )

        # Sort final beam best-first
        beams_sorted = sorted(
            beam.items(),
            key=lambda x: _log_add(x[1][0], x[1][1]),
            reverse=True
        )

        if return_beams:
            return [(list(prefix), _log_add(p_b, p_nb))
                    for prefix, (p_b, p_nb) in beams_sorted]

        best_prefix, _ = beams_sorted[0]
        return self._ids_to_text(list(best_prefix))

    # -----------------------------------------------------------------------
    # Task 4: Beam search with shallow LM fusion
    # -----------------------------------------------------------------------

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Score at each step:
            score = log_p_acoustic + alpha * log_p_lm + beta * num_completed_words

        The LM is scored incrementally: each time a word delimiter token is
        appended we extract the completed word and query KenLM.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)  # (T, V)
        T, V = log_probs.shape
        blank = self.blank_token_id
        wdel = self.word_delimiter_id
        NEG_INF = float('-inf')

        def _init_lm_state() -> kenlm.State:
            s = kenlm.State()
            self.lm_model.BeginSentenceWrite(s)
            return s

        def _tokens_to_last_word(prefix: tuple) -> str:
            """Extract the word just completed (chars since last delimiter)."""
            chars = []
            for tok in reversed(prefix[:-1]):   # skip the trailing delimiter
                if tok == wdel:
                    break
                chars.append(self.vocab[tok])
            return ''.join(reversed(chars)).lower()

        init_state = _init_lm_state()
        beam = {(): (0.0, NEG_INF, 0.0, 0, init_state)}

        for t in range(T):
            lp = log_probs[t]
            new_beam: dict = {}

            def _update(prefix, p_b, p_nb, lm_score, n_words, lm_state):
                combined = _log_add(p_b, p_nb) + self.alpha * lm_score + self.beta * n_words
                if prefix in new_beam:
                    old_combined = (_log_add(new_beam[prefix][0], new_beam[prefix][1])
                                    + self.alpha * new_beam[prefix][2]
                                    + self.beta * new_beam[prefix][3])
                    if combined > old_combined:
                        new_beam[prefix] = (p_b, p_nb, lm_score, n_words, lm_state)
                else:
                    new_beam[prefix] = (p_b, p_nb, lm_score, n_words, lm_state)

            for prefix, (p_b, p_nb, lm_score, n_words, lm_state) in beam.items():
                p_total = _log_add(p_b, p_nb)

                # Emit blank
                _update(prefix, p_total + lp[blank].item(), NEG_INF,
                        lm_score, n_words, lm_state)

                for c in range(V):
                    if c == blank:
                        continue
                    lpc = lp[c].item()

                    if prefix and prefix[-1] == c:
                        _update(prefix, NEG_INF, p_nb + lpc,
                                lm_score, n_words, lm_state)
                        new_prefix = prefix + (c,)
                        new_lm_score, new_n_words, new_lm_state = lm_score, n_words, lm_state
                        if c == wdel and prefix:
                            word = _tokens_to_last_word(new_prefix)
                            if word:
                                delta, new_lm_state = self._lm_score_word(word, lm_state)
                                new_lm_score = lm_score + delta
                                new_n_words = n_words + 1
                        _update(new_prefix, NEG_INF, p_b + lpc,
                                new_lm_score, new_n_words, new_lm_state)
                    else:
                        new_prefix = prefix + (c,)
                        new_lm_score, new_n_words, new_lm_state = lm_score, n_words, lm_state
                        if c == wdel and prefix:
                            word = _tokens_to_last_word(new_prefix)
                            if word:
                                delta, new_lm_state = self._lm_score_word(word, lm_state)
                                new_lm_score = lm_score + delta
                                new_n_words = n_words + 1
                        _update(new_prefix, NEG_INF, p_total + lpc,
                                new_lm_score, new_n_words, new_lm_state)

            def _score(item):
                _, (p_b, p_nb, lm_sc, nw, _state) = item
                return _log_add(p_b, p_nb) + self.alpha * lm_sc + self.beta * nw

            beam = dict(
                sorted(new_beam.items(), key=_score, reverse=True)[:self.beam_width]
            )

        best_prefix = max(beam.items(), key=lambda x: (
            _log_add(x[1][0], x[1][1]) + self.alpha * x[1][2] + self.beta * x[1][3]
        ))[0]
        return self._ids_to_text(list(best_prefix))

    # -----------------------------------------------------------------------
    # Task 6: Second-pass LM rescoring
    # -----------------------------------------------------------------------

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Each hypothesis is rescored with:
            score = log_p_acoustic + alpha * log_p_lm + beta * num_words

        The LM probability is computed over the full sentence at once.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        best_score, best_text = float('-inf'), ""
        for token_ids, log_p_acoustic in beams:
            text = self._ids_to_text(token_ids)
            n_words = len(text.split()) if text else 0
            lm_log_prob = self._lm_score_sentence(token_ids)
            score = log_p_acoustic + self.alpha * lm_log_prob + self.beta * n_words
            if score > best_score:
                best_score, best_text = score, text

        return best_text


    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op. 
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )



def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
