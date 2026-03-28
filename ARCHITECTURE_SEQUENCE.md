1. Speech Input
- File: `inference.py`
- Code:
  - `record_audio(...)`
- Kaam:
  - User ka spoken question record hota hai.


2. Spectrogram / Speech Features
- Files: `inference.py`, `train_stage1.py`
- Code:
  - `audio_to_log_mel(...)`
  - `load_log_mel(...)`
- Kaam:
  - Audio ko log-mel spectrogram features me badla jata hai.


3. Speech Encoder
- File: `speech_encoder.py`
- Code:
  - `SpeechEncoder`
  - `SpeechEncoder.forward(...)`
- Kaam:
  - Speech features ko encoded hidden representations me badalta hai.


4. Connector / Projection
- File: `model.py`
- Code:
  - `ProjectionLayer`
  - `encode_prompt(...)`
- Kaam:
  - Encoder output ko GPT-2 / LLM embedding space me map karta hai.


5. LLM
- File: `model.py`
- Code:
  - `self.lm = AutoModelForCausalLM.from_pretrained(...)`
  - `_run_transformer(...)`
- Kaam:
  - Speech-conditioned tokens ke basis par text/acoustic reasoning karta hai.


6. Text Continuation / Question Understanding
- Files: `model.py`, `train_stage1.py`, `inference.py`
- Code:
  - `generate_question_text(...)`
  - `generate_answer_from_question_text(...)`
  - `build_stage1_text_targets(...)`
- Kaam:
  - Spoken prompt se text-side understanding aur answer text nikalna.


7. Acoustic Continuation
- Files: `model.py`, `train_stage1.py`, `inference.py`
- Code:
  - `forward(...) -> acoustic_output`
  - `reconstruction_loss(...)`
- Kaam:
  - LLM hidden states se answer ki acoustic side nikalna.


8. PreNet
- File: `model.py`
- Code:
  - `PreNet`
- Kaam:
  - Training me ground-truth continuation spectrogram ko LLM-side tokens me badalna.


9. PostNet
- File: `model.py`
- Code:
  - `PostNet`
- Kaam:
  - Hidden states ko spectrogram space me map karna.


10. Spectrogram Output
- File: `model.py`
- Code:
  - `acoustic_output = self.postnet(...)`
- Kaam:
  - Final spectrogram-like acoustic output generate hota hai.


11. Vocoder
- File: `vocoder.py`
- Code:
  - `griffin_lim_vocoder(...)`
- Kaam:
  - Spectrogram ko waveform/audio me convert karta hai.


12. Speech Output
- Files: `inference.py`, `vocoder.py`
- Code:
  - `synthesize_response_audio(...)`
  - `save_waveform(...)`
  - `play_audio(...)`
- Kaam:
  - Final answer audio save/play hoti hai.

Training Sequence
- Stage 1: `train_stage1.py`
- Stage 2: `train_stage2.py`


Current Important Notes
- Step 6 question understanding weak hai.
- Step 7 acoustic branch weak hai.
- Retrieval fallback `inference.py` me use ho raha hai taaki known spoken questions par better answer mil sake.
