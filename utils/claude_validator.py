import json
from typing import Dict, List, Any
import boto3

class ClaudeValidator:
    def __init__(self, region_name: str = "us-east-1", profile_name: str = None):
        """
        Initialize Claude validator using AWS Bedrock with IAM roles.
        """
        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        self.client = session.client("bedrock-runtime", region_name=region_name)
        self.model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        
    def _invoke_claude(self, messages: List[Dict], max_tokens: int = 4000, temperature: float = 0.1) -> str:
        """
        Helper method to invoke Claude via Bedrock.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
        }
        
        try:
            print(f"Invoking Claude with model: {self.model_id}")
            print(f"Request body: {json.dumps(body, indent=2)}")
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response["body"].read())
            print(f"Raw Claude response: {json.dumps(response_body, indent=2)}")
            
            if "content" in response_body and len(response_body["content"]) > 0:
                response_text = response_body["content"][0]["text"]
                print(f"Extracted text length: {len(response_text)}")
                print(f"First 500 chars: {response_text[:500]}")
                return response_text
            else:
                raise Exception(f"No content in Claude response. Full response: {response_body}")
                
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Bedrock response as JSON: {str(e)}")
        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")
        
    def validate_application_against_rules(
        self, 
        application_text: str, 
        rules_text: str,
        claim_type: str = "insurance claim"
    ) -> Dict[str, Any]:
        """
        Use Claude 3.7 via Bedrock to compare application against rules and return validation results.
        """
        
        validation_prompt = f"""
You are an expert document reviewer specializing in {claim_type} validation. Your task is to carefully compare a submitted application against a comprehensive set of rules and requirements.

<rules_document>
{rules_text}
</rules_document>

<submitted_application>
{application_text}
</submitted_application>

Please analyze the submitted application against the rules document and provide a comprehensive validation report in the following JSON format:

{{
    "overall_status": "APPROVED" or "REJECTED" or "NEEDS_REVIEW",
    "confidence_score": 0.0 to 1.0,
    "compliance_summary": {{
        "total_requirements_checked": number,
        "requirements_met": number,
        "requirements_failed": number,
        "requirements_unclear": number
    }},
    "detailed_findings": [
        {{
            "requirement": "specific requirement from rules",
            "status": "MET" or "FAILED" or "UNCLEAR" or "NOT_APPLICABLE",
            "evidence": "specific text from application that addresses this requirement",
            "explanation": "detailed explanation of why this requirement is met/failed/unclear",
            "severity": "CRITICAL" or "MAJOR" or "MINOR"
        }}
    ],
    "missing_documents": [
        "list of required documents that appear to be missing"
    ],
    "missing_information": [
        "list of required information that appears to be missing"
    ],
    "recommendations": [
        "specific actionable recommendations to address issues"
    ],
    "additional_notes": "any other relevant observations or concerns"
}}

Guidelines for your analysis:
1. Be thorough and systematic - check every requirement mentioned in the rules
2. Look for both explicit compliance (directly stated) and implicit compliance (can be reasonably inferred)
3. Pay attention to document formatting, required fields, signatures, dates, and specific procedural requirements
4. Consider both mandatory requirements (that would cause rejection) and best practices
5. If information is ambiguous or unclear in either document, note this explicitly
6. Pay special attention to EXIM Bank specific requirements and procedures
7. Provide specific evidence from the application text when possible

Provide your response as valid JSON only, with no additional text or explanation outside the JSON structure.
"""

        try:
            messages = [{"role": "user", "content": validation_prompt}]
            response_text = self._invoke_claude(messages, max_tokens=4000, temperature=0.1)
            
            print(f"Claude response text: {response_text[:1000]}...")  # Debug output
            
            # Clean up the response text
            response_text = response_text.strip()
            
            # Try to find JSON in the response if it's wrapped in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                print(f"Extracted JSON: {json_text[:500]}...")
                result = json.loads(json_text)
                return result
            else:
                # If no JSON found, try to parse the entire response
                result = json.loads(response_text)
                return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Response that failed to parse: {response_text if 'response_text' in locals() else 'No response'}")
            
            # Return a structured error response instead of failing
            return {
                "overall_status": "ERROR",
                "error": f"Failed to parse Claude response as JSON: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else "No response",
                "suggestions": [
                    "The document may be too complex for automated processing",
                    "Try submitting a clearer, more structured document",
                    "Manual review may be required"
                ]
            }
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {
                "overall_status": "ERROR", 
                "error": f"Claude validation error: {str(e)}",
                "suggestions": [
                    "Check AWS Bedrock permissions",
                    "Verify the model is available in your region",
                    "Try again later if this is a temporary service issue"
                ]
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the Bedrock connection with a simple query.
        """
        try:
            messages = [{"role": "user", "content": "Hello, please respond with 'Connection successful'"}]
            response = self._invoke_claude(messages, max_tokens=50, temperature=0.1)
            
            return {
                "status": "connected",
                "model": self.model_id,
                "response": response
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }