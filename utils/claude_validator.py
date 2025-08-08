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
        
    def validate_multiple_documents(self, claim_type, rules_text, documents_dict):
        """
        Validate multiple documents against rules and return individual + overall reports
        
        Args:
            claim_type: Type of claim being validated
            rules_text: The rules document text
            documents_dict: Dictionary where keys are document names/types and values are document content
        """
        
        validation_prompt = f"""
    You are an expert document reviewer specializing in {claim_type} validation. Your task is to carefully analyze multiple submitted documents against a comprehensive set of rules and requirements.

    <rules_document>
    {rules_text}
    </rules_document>

    <submitted_documents>
    {self._format_documents_for_prompt(documents_dict)}
    </submitted_documents>

    Please analyze each submitted document individually against the rules document, then provide an overall assessment. Provide your response in the following JSON format:

    {{
        "individual_document_reports": [
            {{
                "document_name": "name/type of the document",
                "document_status": "APPROVED" or "REJECTED" or "NEEDS_REVIEW",
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
                        "evidence": "specific text from document that addresses this requirement",
                        "explanation": "detailed explanation of why this requirement is met/failed/unclear",
                        "severity": "CRITICAL" or "MAJOR" or "MINOR"
                    }}
                ],
                "missing_information": [
                    "list of required information missing from this specific document"
                ],
                "recommendations": [
                    "specific actionable recommendations for this document"
                ],
                "document_specific_notes": "observations specific to this document"
            }}
        ],
        "overall_assessment": {{
            "overall_status": "APPROVED" or "REJECTED" or "NEEDS_REVIEW",
            "overall_confidence_score": 0.0 to 1.0,
            "cross_document_compliance": {{
                "total_documents_analyzed": number,
                "documents_approved": number,
                "documents_rejected": number,
                "documents_needing_review": number
            }},
            "missing_documents": [
                "list of required documents that are completely missing from submission"
            ],
            "cross_document_inconsistencies": [
                {{
                    "issue": "description of inconsistency between documents",
                    "affected_documents": ["list of document names with the inconsistency"],
                    "severity": "CRITICAL" or "MAJOR" or "MINOR",
                    "recommendation": "how to resolve the inconsistency"
                }}
            ],
            "overall_recommendations": [
                "high-level actionable recommendations for the entire submission"
            ],
            "completeness_assessment": {{
                "all_required_documents_present": true or false,
                "all_required_information_present": true or false,
                "ready_for_processing": true or false
            }},
            "additional_notes": "any other relevant observations about the overall submission"
        }}
    }}

    Guidelines for your analysis:
    1. INDIVIDUAL DOCUMENT ANALYSIS:
    - Analyze each document separately against applicable requirements
    - Focus on what that specific document should contain according to the rules
    - Note if a document is well-formatted and complete for its type
    - Don't penalize a document for not containing information that should be in a different document

    2. CROSS-DOCUMENT ANALYSIS:
    - Look for consistency in information across documents (dates, names, amounts, etc.)
    - Identify if required documents are missing entirely
    - Check if the combination of documents meets all overall requirements
    - Verify that documents reference each other correctly where required

    3. DECISION LOGIC:
    - Individual documents can be APPROVED even if other documents have issues
    - Overall status should reflect the weakest link in the chain
    - If any critical requirements are unmet across all documents, overall status should be REJECTED
    - Use NEEDS_REVIEW when there are ambiguities or minor issues that need human judgment

    4. GENERAL GUIDELINES:
    - Be thorough and systematic - check all requirements mentioned in the rules
    - Look for both explicit compliance (directly stated) and implicit compliance (reasonably inferred)
    - Pay attention to formatting, required fields, signatures, dates, and procedural requirements
    - If information is ambiguous, note this explicitly
    - Pay special attention to {claim_type} specific requirements and procedures
    - Provide specific evidence from documents when possible
    - If unsure, lean toward accepting (false positives better than false negatives) as accepted applications will be manually reviewed
    - Ignore timeliness requirements that cannot be verified from document content alone

    Provide your response as valid JSON only, with no additional text or explanation outside the JSON structure.
    """

        try:
            messages = [{"role": "user", "content": validation_prompt}]
            response_text = self._invoke_claude(messages, max_tokens=6000, temperature=0.1)
            
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
                "individual_document_reports": [],
                "overall_assessment": {
                    "overall_status": "ERROR",
                    "error": f"Failed to parse Claude response as JSON: {str(e)}",
                    "raw_response": response_text if 'response_text' in locals() else "No response",
                    "suggestions": [
                        "The documents may be too complex for automated processing",
                        "Try submitting clearer, more structured documents",
                        "Manual review may be required"
                    ]
                }
            }
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {
                "individual_document_reports": [],
                "overall_assessment": {
                    "overall_status": "ERROR", 
                    "error": f"Claude validation error: {str(e)}",
                    "suggestions": [
                        "Check AWS Bedrock permissions",
                        "Verify the model is available in your region",
                        "Try again later if this is a temporary service issue"
                    ]
                }
            }

    def _format_documents_for_prompt(self, documents_dict):
        """Helper method to format multiple documents for the prompt"""
        formatted_docs = ""
        for doc_name, doc_content in documents_dict.items():
            formatted_docs += f"""
    <document name="{doc_name}">
    {doc_content}
    </document>
    """
        return formatted_docs.strip()
        

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