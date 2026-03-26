"""
Sentinel AI - Audio Analyzer
Built-in speech-to-text using faster-whisper (local, no API key needed),
then content analysis via Google Gemini.
"""
import os
from pathlib import Path
from typing import Dict


class AudioAnalyzer:
    """
    Audio analysis using local Whisper STT + Gemini for scam/deepfake detection.
    """

    def __init__(self):
        from faster_whisper import WhisperModel
        # Use tiny model for speed on CPU (free tier friendly)
        self.whisper = WhisperModel("tiny", device="cpu", compute_type="int8")
        self.loaded = True

    def _transcribe(self, file_path: Path) -> tuple[str, float, float]:
        """
        Transcribe audio file using local Whisper model.
        Returns (transcription, avg_confidence, duration_seconds)
        """
        segments, info = self.whisper.transcribe(str(file_path), beam_size=1)
        
        text_parts = []
        confidences = []
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            # avg_logprob is negative; convert to 0-1 confidence
            conf = min(1.0, max(0.0, 1.0 + segment.avg_logprob))
            confidences.append(conf)

        transcription = " ".join(text_parts).strip()
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        duration = info.duration

        return transcription, avg_confidence, duration

    def _analyze_with_gemini(self, text: str, confidence: float) -> tuple[str, int, str]:
        """
        Analyze transcribed text with Gemini for spam/scam detection.
        Returns (verdict, risk_score, reasoning)
        """
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""Analyze this audio transcription and determine if it's a scam, spam, or legitimate.

Transcription: "{text}"
Speech confidence score: {confidence:.2f}

Check for:
- Scam/phishing language (urgency, money requests, threats, prizes)
- Impersonation (bank, government, tech support)
- Social engineering tactics
- Robocall/spam patterns
- Suspicious requests for personal info

Respond in this exact format:
VERDICT: [Scam/Spam/Suspicious/Legitimate]
RISK_SCORE: [0-100]
REASONING: [One sentence explanation]"""

        response = model.generate_content(prompt)
        analysis = response.text

        verdict = "Legitimate"
        risk_score = 20
        reasoning = "No suspicious content detected."

        if "VERDICT:" in analysis:
            line = [l for l in analysis.split("\n") if "VERDICT:" in l][0]
            verdict = line.split("VERDICT:")[1].strip().split()[0]

        if "RISK_SCORE:" in analysis:
            line = [l for l in analysis.split("\n") if "RISK_SCORE:" in l][0]
            try:
                risk_score = int("".join(filter(str.isdigit, line.split("RISK_SCORE:")[1][:5])))
            except:
                risk_score = 50

        if "REASONING:" in analysis:
            reasoning = analysis.split("REASONING:")[1].strip()

        return verdict, risk_score, reasoning

    def analyze(self, file_path: Path) -> Dict:
        try:
            # Step 1: Transcribe locally
            transcription, confidence, duration = self._transcribe(file_path)

            # Step 2: Analyze with Gemini if we got text
            if transcription:
                verdict, risk_score, reasoning = self._analyze_with_gemini(transcription, confidence)
            else:
                verdict, risk_score, reasoning = "Suspicious", 60, "Could not transcribe audio clearly."

            # Low confidence = possibly synthetic/manipulated voice
            if confidence < 0.5:
                risk_score = min(100, risk_score + 15)
                reasoning += " Low speech clarity may indicate synthetic voice."

            r = risk_score / 100.0
            v = verdict.upper()

            if "SCAM" in v or "SPAM" in v:
                human_voice = 1.0 - r
                tts_likelihood = r * 0.6
                voice_cloning = r * 0.4
            elif "SUSPICIOUS" in v:
                human_voice = 1.0 - r
                tts_likelihood = r * 0.4
                voice_cloning = r * 0.3
            else:
                human_voice = 1.0 - (r * 0.3)
                tts_likelihood = r * 0.2
                voice_cloning = r * 0.1

            return {
                "human_voice": human_voice,
                "tts_likelihood": tts_likelihood,
                "voice_cloning": voice_cloning,
                "reasoning": reasoning,
                "transcription": transcription,
                "confidence": confidence,
                "duration_seconds": duration,
                "risk_score": risk_score
            }

        except Exception as e:
            print(f"Audio analysis failed: {e}")
            return {
                "human_voice": 0.5,
                "tts_likelihood": 0.25,
                "voice_cloning": 0.25,
                "reasoning": f"Analysis failed: {str(e)}",
                "transcription": "",
                "confidence": 0.0,
                "duration_seconds": 0.0,
                "risk_score": 50
            }


_analyzer = None


def get_audio_analyzer() -> AudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer()
    return _analyzer
