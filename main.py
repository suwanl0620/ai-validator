from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import uuid
import os
import tempfile
from utils.s3 import download_pdf_from_s3
from utils.pdf_extractor import extract_text_from_pdf
from utils.claude_validator import ClaudeValidator

app = FastAPI(
    title="EXIM Bank Claim Validation API",
    description="AI-powered claim validation using Claude 3.7 via AWS Bedrock",
    version="1.0.0"
)

# Configuration
S3_BUCKET = "ils-rule-documents"
RULES_KEY = "EXIM Rules for AI Model.pdf"

# Initialize Claude validator using Bedrock (no API key needed with IAM roles)
claude_validator = ClaudeValidator(region_name="us-east-1", profile_name="mainils") #change profile name to None

# Cache for rules to avoid repeated S3 downloads
_cached_rules_text = None

async def get_rules_text() -> str:
    """Get rules text, using cache if available"""
    global _cached_rules_text
    
    if _cached_rules_text is None:
        temp_dir = tempfile.gettempdir()
        rules_path = os.path.join(temp_dir, f"exim_rules_{uuid.uuid4()}.pdf")
        try:
            download_pdf_from_s3(S3_BUCKET, RULES_KEY, rules_path, profile_name="mainils") #change profile name to None
            _cached_rules_text = extract_text_from_pdf(rules_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download or parse rules: {str(e)}")
        finally:
            if os.path.exists(rules_path):
                os.remove(rules_path)
    
    return _cached_rules_text

@app.post("/submit-claim/")
async def submit_claim(files: List[UploadFile] = File(..., description="PDF files containing the claim application (up to 5 files)")):
    """
    Submit and validate a claim using Claude 3.7 via Bedrock for intelligent comparison.
    
    Upload up to 5 PDF files and get back a detailed validation report comparing them against EXIM Bank rules.
    """
    user_form_paths = []
    
    try:
        # Validate number of files
        if len(files) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="At least 1 file is required")
        
        documents_dict = {}
        temp_dir = tempfile.gettempdir()
        
        # Process each uploaded file
        for i, file in enumerate(files):
            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"Only PDF files are accepted. File '{file.filename}' is not a PDF")
            
            # Check file size
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=400, detail=f"File '{file.filename}' too large. Maximum size is 10MB")
            
            # Save uploaded file
            user_form_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
            user_form_paths.append(user_form_path)
            with open(user_form_path, "wb") as f:
                f.write(contents)
            
            # Extract text from file
            form_text = extract_text_from_pdf(user_form_path)
            
            if not form_text.strip():
                raise HTTPException(status_code=400, detail=f"Could not extract text from '{file.filename}'. Please ensure the PDF contains readable text.")
            
            # Add to documents dictionary
            documents_dict[file.filename] = form_text

        # Get rules text
        rules_text = await get_rules_text()

        # Use Claude 3.7 to validate the application against rules
        validation_result = claude_validator.validate_multiple_documents(
            claim_type="EXIM Bank insurance claim",
            rules_text=rules_text,
            documents_dict=documents_dict
        )
        
        # Handle validation errors
        if validation_result.get("overall_assessment", {}).get("overall_status") == "ERROR":
            error_msg = validation_result.get("overall_assessment", {}).get("error", "Unknown validation error")
            suggestions = validation_result.get("overall_assessment", {}).get("suggestions", [])
            
            return {
                "status": "error",
                "error": error_msg,
                "suggestions": suggestions,
                "raw_response": validation_result.get("overall_assessment", {}).get("raw_response", "")[:500] if validation_result.get("overall_assessment", {}).get("raw_response") else ""
            }
        
        # Prepare response
        status_mapping = {
            "APPROVED": "approved",
            "REJECTED": "rejected", 
            "NEEDS_REVIEW": "needs_review"
        }
        
        overall_assessment = validation_result.get("overall_assessment", {})
        status = status_mapping.get(overall_assessment.get("overall_status"), "needs_review")
        
        # Extract issues from individual document reports
        issues = []
        critical_issues = []
        all_recommendations = []
        
        for doc_report in validation_result.get("individual_document_reports", []):
            for finding in doc_report.get("detailed_findings", []):
                if finding.get("status") == "FAILED":
                    issue_text = f"{doc_report.get('document_name', 'Unknown document')} - {finding.get('requirement', 'Unknown requirement')}: {finding.get('explanation', 'No explanation')}"
                    if finding.get("severity") == "CRITICAL":
                        critical_issues.append(issue_text)
                    else:
                        issues.append(issue_text)
            
            # Add document-specific recommendations
            all_recommendations.extend(doc_report.get("recommendations", []))
        
        # Add overall recommendations
        all_recommendations.extend(overall_assessment.get("overall_recommendations", []))
        
        # Add missing documents and cross-document issues
        missing_docs = overall_assessment.get("missing_documents", [])
        cross_doc_issues = overall_assessment.get("cross_document_inconsistencies", [])
        
        issues.extend([f"Missing document: {doc}" for doc in missing_docs])
        issues.extend([f"Cross-document issue: {issue.get('issue', 'Unknown issue')}" for issue in cross_doc_issues])
        
        return {
            "status": status,
            "confidence_score": overall_assessment.get("overall_confidence_score", 0.0),
            "critical_issues": critical_issues,
            "issues": issues,
            "compliance_summary": overall_assessment.get("cross_document_compliance", {}),
            "recommendations": all_recommendations,
            "additional_notes": overall_assessment.get("additional_notes", ""),
            "individual_document_reports": validation_result.get("individual_document_reports", []),
            "completeness_assessment": overall_assessment.get("completeness_assessment", {}),
            "model_used": "Claude 3.7 Sonnet via AWS Bedrock",
            "filenames": [file.filename for file in files]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary files
        for user_form_path in user_form_paths:
            if os.path.exists(user_form_path):
                os.remove(user_form_path)

@app.get("/test-claude")
async def test_claude():
    """
    Simple test endpoint to verify Claude 3.7 connectivity via AWS Bedrock.
    
    Returns connection status and a simple test response from Claude.
    """
    try:
        test_result = claude_validator.test_connection()
        return {
            "claude_status": test_result.get("status", "unknown"),
            "model": "Claude 3.7 Sonnet via AWS Bedrock",
            "region": "us-east-1",
            "test_response": test_result.get("response", ""),
            "error": test_result.get("error") if test_result.get("status") == "error" else None
        }
    except Exception as e:
        return {
            "claude_status": "error",
            "error": str(e),
            "model": "Claude 3.7 Sonnet via AWS Bedrock"
        }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify system status.
    
    Checks Claude API connectivity, S3 access, and overall system health.
    """
    health_status = {
        "status": "healthy",
        "services": {},
        "timestamp": uuid.uuid4().hex
    }
    
    # Test Claude connection
    try:
        claude_test = claude_validator.test_connection()
        health_status["services"]["claude"] = {
            "status": "healthy" if claude_test["status"] == "connected" else "unhealthy",
            "model": "Claude 3.7 Sonnet via AWS Bedrock",
            "details": claude_test.get("response", claude_test.get("error", ""))
        }
    except Exception as e:
        health_status["services"]["claude"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Test S3 access
    try:
        rules_text = await get_rules_text()
        health_status["services"]["s3"] = {
            "status": "healthy",
            "rules_length": len(rules_text),
            "bucket": S3_BUCKET,
            "key": RULES_KEY
        }
    except Exception as e:
        health_status["services"]["s3"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Overall status
    if any(service.get("status") == "unhealthy" for service in health_status["services"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status