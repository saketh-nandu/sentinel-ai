"""
Sentinel AI - Text Analysis Route
POST /analyze/text endpoint
"""
from fastapi import APIRouter, HTTPException

from app.schemas.responses import (
    TextAnalysisRequest,
    TextAnalysisResult,
    TextAnalysisDetails,
    ErrorResponse
)
from app.models.text_analyzer import get_text_analyzer
from app.utils.explainer import explain_text_analysis, get_verdict


router = APIRouter()


@router.post(
    "/text",
    response_model=TextAnalysisResult,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    summary="Analyze text for AI generation and scam intent",
    description="Analyzes text content to detect if it's AI-generated and check for scam/fraud indicators."
)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze text content for:
    - AI-generated content
    - Scam/phishing intent
    - Urgency manipulation
    - Financial requests
    - Impersonation attempts
    """
    try:
        # Get analyzer
        analyzer = get_text_analyzer()
        
        # Run analysis
        scores = analyzer.analyze(request.text)
        
        # Map analyzer output to expected keys
        ai_likelihood = scores.get("ai_generated", 0.0)
        scam_intent = scores.get("scam_probability", 0.0)
        risk_score_raw = scores.get("risk_score", 50)
        
        # Generate explanations
        risk_score, explanations, action = explain_text_analysis(
            ai_likelihood=ai_likelihood,
            scam_intent=scam_intent,
            urgency=scam_intent * 0.5,
            financial=scam_intent * 0.4,
            impersonation=scam_intent * 0.3
        )
        
        # Use Gemini's risk score if available
        if risk_score_raw:
            risk_score = risk_score_raw
        
        # Build response
        return TextAnalysisResult(
            risk_score=risk_score,
            verdict=get_verdict(risk_score),
            explanations=explanations,
            action=action,
            content_type="text",
            details=TextAnalysisDetails(
                ai_likelihood=ai_likelihood,
                scam_intent=scam_intent,
                urgency_level=scam_intent * 0.5,
                financial_request=scam_intent * 0.4,
                impersonation=scam_intent * 0.3
            )
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
