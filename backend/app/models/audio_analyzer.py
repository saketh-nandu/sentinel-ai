"""
Sentinel AI - Audio Analyzer
Uses Google Gemini's native audio understanding for transcription + analysis.
No external STT API needed, no large model downloads.
"""
import os
from pathlib import Path
from typing import Dict


class AudioAnalyzer:

    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.loaded = True

    def analyze(self, file_path: Path) -> Dict:
        try:
            import google.generativeai as genai

            # Upload audio file to Gemini
            audio_file = genai.upload_file(str(file_path))

            prompt = """Listen to this audio and do two things:

1. Transcribe exactly what is being said.
2. Analyze if it's a scam, spam, robocall, or legitimate.

Check for: urgency tactics, money requests, threats, impersonation (bank/govt/tech support), prize claims, suspicious requests.

Respond in this exact format:
TRANSCRIPTION: [exact words spoken]
VERDICT: [Scam/Spam/Suspicious/Legitimate]
RISK_SCORE: [0-100]
REASONING: [one sentence explanation]"""

            response = self.model.generate_content([prompt, audio_file])
            analysis = response.text

            # Parse transcription
            transcription = ""
            if "TRANSCRIPTION:" in analysis:
                t_line = analysis.split("TRANSCRIPTION:")[1]
                transcription = t_line.split("\n")[0].strip()

            # Parse verdict
            verdict = "Legitimate"
            if "VERDICT:" in analysis:
                line = [l for l in analysis.split("\n") if "VERDICT:" in l][0]
                verdict = line.split("VERDICT:")[1].strip().split()[0]

            # Parse risk score
            risk_score = 20
            if "RISK_SCORE:" in analysis:
                line = [l for l in analysis.split("\n") if "RISK_SCORE:" in l][0]
                try:
                    risk_score = int("".join(filter(str.isdigit, line.split("RISK_SCORE:")[1][:5])))
                except:
                    risk_score = 50

            # Parse reasoning
            reasoning = "No suspicious content detected."
            if "REASONING:" in analysis:
                reasoning = analysis.split("REASONING:")[1].strip()

            # Get duration via soundfile
            try:
                import soundfile as sf
                info = sf.info(str(file_path))
                duration = info.duration
            except:
                duration = 0.0

            r = risk_score / 100.0
            v = verdict.upper()

            if "SCAM" in v or "SPAM" in v:
                return {
                    "human_voice": 1.0 - r,
                    "tts_likelihood": r * 0.6,
                    "voice_cloning": r * 0.4,
                    "reasoning": reasoning,
                    "transcription": transcription,
                    "confidence": 0.9,
                    "duration_seconds": duration,
                    "risk_score": risk_score
                }
            elif "SUSPICIOUS" in v:
                return {
                    "human_voice": 1.0 - r,
                    "tts_likelihood": r * 0.4,
                    "voice_cloning": r * 0.3,
                    "reasoning": reasoning,
                    "transcription": transcription,
                    "confidence": 0.8,
                    "duration_seconds": duration,
                    "risk_score": risk_score
                }
            else:
                return {
                    "human_voice": 1.0 - (r * 0.3),
                    "tts_likelihood": r * 0.2,
                    "voice_cloning": r * 0.1,
                    "reasoning": reasoning,
                    "transcription": transcription,
                    "confidence": 0.95,
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
