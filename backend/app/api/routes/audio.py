"""
Sentinel AI - Audio Analysis Route
POST /analyze/audio endpoint
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks

from app.schemas.responses import AudioAnalysisResult, AudioAnalysisDetails, ErrorResponse
from app.models.audio_analyzer import get_audio_analyzer
from app.utils.file_handler import save_upload, delete_file
from app.utils.explainer import explain_audio_analysis, get_verdict


router = APIRouter()


@router.post(
    "/audio",
    response_model=AudioAnalysisResult,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze audio for voice spoofing and scam content",
    description="Transcribes audio using built-in Whisper STT then analyzes content for scams, deepfakes, and voice cloning."
)
async def analyze_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file (MP3, WAV, M4A)")
):
    file_path = None

    try:
        file_path, file_id = await save_upload(file, "audio")

        analyzer = get_audio_analyzer()
        result = analyzer.analyze(file_path)

        background_tasks.add_task(delete_file, file_path)

        risk_score, explanations, action = explain_audio_analysis(
            human_voice=result["human_voice"],
            tts_likelihood=result["tts_likelihood"],
            voice_cloning=result["voice_cloning"]
        )

        # Add transcription to explanations if available
        transcription = result.get("transcription", "")
        if transcription:
            explanations = [f'Transcribed: "{transcription[:200]}"'] + explanations
            explanations = explanations[:3]

        return AudioAnalysisResult(
            risk_score=risk_score,
            verdict=get_verdict(risk_score),
            explanations=explanations,
            action=action,
            content_type="audio",
            details=AudioAnalysisDetails(
                human_voice=result["human_voice"],
                tts_likelihood=result["tts_likelihood"],
                voice_cloning=result["voice_cloning"]
            ),
            duration_seconds=result.get("duration_seconds", 0.0)
        )

    except HTTPException:
        if file_path:
            delete_file(file_path)
        raise
    except Exception as e:
        if file_path:
            delete_file(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
