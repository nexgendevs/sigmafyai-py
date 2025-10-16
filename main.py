# Third-party imports
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd
import traceback

# Local imports
import utils

import os

# Load environment variables
load_dotenv()

app = Flask(__name__)


current_environment = os.getenv('ENVIRONMENT', 'production')  # Get the actual value

@app.route("/")
def index():
    return jsonify({"message": f"AI Sigmafy running in {current_environment} mode..."})

@app.route('/api/generate-stat-graph', methods=['POST'])
def generate_stat_graph():
    try:
        print("=== Starting generate_stat_graph request ===")
        print(f"🔍 Request Headers: {dict(request.headers)}")
        print(f"🔍 Request Form Data: {dict(request.form)}")
        print(f"🔍 Request Method: {request.method}")

        # Validate request method
        if request.method != 'POST':
            return jsonify({
                "success": False,
                "error": "Method not allowed. Use POST.",
                "details": f"Received {request.method}, expected POST"
            }), 405

        # Validate required fields
        if not request.form:
            return jsonify({
                "success": False,
                "error": "No form data received",
                "details": "Request must contain form data with csv_url and graph_type"
            }), 400

        if 'csv_url' not in request.form or 'graph_type' not in request.form:
            missing_fields = []
            if 'csv_url' not in request.form:
                missing_fields.append('csv_url')
            if 'graph_type' not in request.form:
                missing_fields.append('graph_type')
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "details": "Required fields: csv_url, graph_type"
            }), 400

        # Safely extract and validate form data
        csv_url = request.form.get('csv_url', '').strip()
        graph_type = request.form.get('graph_type', '').strip().lower()
        chart_generation_prompt = request.form.get('chart_generation_prompt', '').strip()

        # Validate URL and graph type are not empty
        if not csv_url:
            return jsonify({
                "success": False,
                "error": "csv_url cannot be empty",
                "details": "Please provide a valid CSV URL"
            }), 400

        if not graph_type:
            return jsonify({
                "success": False,
                "error": "graph_type cannot be empty",
                "details": "Please specify a chart type"
            }), 400

        print(f"📋 Parameters received:")
        print(f"   CSV URL: {csv_url}")
        print(f"   Graph Type: {graph_type}")
        print(f"   Custom Prompt: {'Yes' if chart_generation_prompt.strip() else 'No'}")

        # # Validate graph type
        # allowed_graph_types = [
        #     "bar chart",
        #     "line chart", 
        #     "scatter plot",
        #     "pie chart",
        #     "histogram",
        #     "particle chart"
        # ]
        
        # if graph_type not in allowed_graph_types:
        #     return jsonify({
        #         "success": False, 
        #         "error": f"Invalid graph type. Allowed types are: {', '.join(allowed_graph_types)}"
        #     }), 400

        # Use robust CSV loading with error handling
        print("📥 Starting CSV loading...")
        try:
            df = utils.load_csv_robust(csv_url)
            if df is None or df.empty:
                return jsonify({
                    "success": False,
                    "error": "CSV file is empty or could not be loaded",
                    "details": f"URL: {csv_url}"
                }), 400
            print(f"✅ CSV loaded successfully. Shape: {df.shape}")
        except Exception as csv_error:
            print(f"❌ CSV loading failed: {str(csv_error)}")
            return jsonify({
                "success": False,
                "error": "Failed to load CSV file",
                "details": f"Error: {str(csv_error)}"
            }), 400

        print("🤖 Starting LLM graph generation...")
        try:
            python_code, token_usage = utils.generate_graph_with_llm(df, graph_type, chart_generation_prompt)
            if not python_code or not python_code.strip():
                return jsonify({
                    "success": False,
                    "error": "AI failed to generate chart code",
                    "details": "The AI service returned empty code"
                }), 500
            print("✅ LLM generation completed")
        except Exception as llm_error:
            print(f"❌ LLM generation failed: {str(llm_error)}")
            return jsonify({
                "success": False,
                "error": "AI chart generation failed",
                "details": str(llm_error)
            }), 500

        print("🐍 Starting Python code execution...")
        try:
            encoded_img, error = utils.execute_python_code(python_code)
            print("✅ Python code execution completed")

            if error:
                return jsonify({
                    "success": False,
                    "error": "Chart generation execution failed",
                    "details": str(error)
                }), 500

            if not encoded_img or not encoded_img.strip():
                return jsonify({
                    "success": False,
                    "error": "Chart generation produced no image",
                    "details": "The code executed but no image was generated"
                }), 500

        except Exception as exec_error:
            print(f"❌ Code execution failed: {str(exec_error)}")
            return jsonify({
                "success": False,
                "error": "Python code execution failed",
                "details": str(exec_error)
            }), 500

        print(f"Generated code:\n\n{python_code}")
        return jsonify({
            "success": True,
            "base64_image": encoded_img,
            "message": "Chart generated successfully",
            "token_usage": token_usage
        })

    except Exception as e:
        print(f"🔥 Unexpected error in generate_stat_graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error occurred",
            "details": str(e),
            "type": "unexpected_error"
        }), 500


@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    try:
        print("=== Starting analyze_document request ===")
        print(f"🔍 Request Headers: {dict(request.headers)}")
        print(f"🔍 Request Form Data Keys: {list(request.form.keys())}")
        print(f"🔍 Request Method: {request.method}")

        # Validate request method
        if request.method != 'POST':
            return jsonify({
                "success": False,
                "error": "Method not allowed. Use POST.",
                "details": f"Received {request.method}, expected POST"
            }), 405
        # Validate form data exists
        if not request.form:
            return jsonify({
                "success": False,
                "error": "No form data received",
                "details": "Request must contain form data"
            }), 400

        # Check for required fields
        required_fields = ['question_title', 'question_content', 'approval_criteria', 'txt_only_submission']
        missing_fields = [field for field in required_fields if field not in request.form or not request.form[field].strip()]

        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing or empty required fields: {', '.join(missing_fields)}",
                "details": f"Required fields: {', '.join(required_fields)}"
            }), 400

        # Get form data
        question_title = request.form['question_title']
        question_content = request.form['question_content']
        approval_criteria = request.form['approval_criteria']
        txt_only_submission = request.form['txt_only_submission']

        # Handle optional document submission via URL
        file_url = None
        file_path = None
        file_id = None

        # Check for mutually exclusive optional fields
        has_document_url = 'document_url' in request.form and request.form['document_url'].strip()
        has_chart_fields = 'chart_csv_url' in request.form and 'chart_type' in request.form

        if has_document_url and has_chart_fields:
            return jsonify({
                "success": False,
                "error": "Cannot provide both document_url and chart fields. Please use either document analysis or chart generation, not both."
            }), 400

        # Handle document analysis
        if has_document_url:
            file_url = request.form['document_url'].strip()

            # Validate submission correctness before analyzing document
            mark_result = utils.mark_answer(question_title, question_content, approval_criteria, txt_only_submission)
            
            utils.logger.info(f"Mark result for text submission:\n\n{mark_result}")

            # Analyze document with vision API
            analysis_result, analysis_error = utils.analyze_document_with_vision(
                file_url, question_title, question_content, approval_criteria
            )

            if analysis_error:
                utils.logger.error(f"Unable to analyze document: '{analysis_error}'")
                return jsonify({
                    "success": False,
                    "error": "Document analysis failed",
                    "details": str(analysis_error)
                }), 500

            # Combine both text submission and document analysis results
            return jsonify({
                "success": True,
                "text_result": {
                    "analysis": {
                        "response": mark_result.get('response', ''),
                        "score": mark_result.get('score', 0),
                        "solution": mark_result.get('solution', '')
                    },
                    "success": mark_result['success']
                },
                "document_result": {
                    "analysis": analysis_result.get('analysis', '') if analysis_result else '',
                    "solution": analysis_result.get('solution', '') if analysis_result else '',
                    "score": analysis_result.get('score', '') if analysis_result else '',
                    "success": analysis_result.get('success', False) if analysis_result else False
                }
            })

        # Handle chart generation
        elif has_chart_fields:
            try:
                chart_csv_url = request.form['chart_csv_url']
                chart_type = request.form['chart_type']
                chart_generation_prompt = request.form.get('chart_generation_prompt', '')

                # Validate submission correctness before generating chart
                mark_result = utils.mark_answer(question_title, question_content, approval_criteria, txt_only_submission)
                if not mark_result['success'] or mark_result['solution'] != 'approved':
                    return jsonify({"success": False, "error": "Incorrect submission. Chart generation aborted."}), 400

                if not chart_csv_url.strip():
                    return jsonify({
                        "success": False,
                        "error": "Chart CSV URL is required when chart_type is provided."
                    }), 400

                # Use robust CSV loading for URLs
                df = utils.load_csv_robust(chart_csv_url)

                python_code, token_usage = utils.generate_graph_with_llm(df, chart_type, chart_generation_prompt)
                base64_image, chart_error = utils.execute_python_code(python_code)

                if chart_error:
                    return jsonify({
                        "success": False,
                        "error": "Chart generation failed",
                        "details": str(chart_error)
                    }), 500

                return jsonify({
                    "success": True,
                    "base64_image": base64_image,
                    "token_usage": token_usage
                })

            except Exception as chart_ex:
                print(f"❌ Chart generation error: {str(chart_ex)}")
                traceback.print_exc()
                return jsonify({
                    "success": False,
                    "error": "Chart generation failed",
                    "details": str(chart_ex)
                }), 500

        # Handle text-only submission (fallback when no optional fields provided)
        else:
            print(f"Fallback to text-only submission...")
            # Handle text-only submission using mark_answer function
            mark_result = utils.mark_answer(question_title, question_content, approval_criteria, txt_only_submission)

            if not mark_result['success']:
                return jsonify({
                    "success": False,
                    "error": "Text analysis failed",
                    "details": mark_result.get('error', 'Unknown error')
                }), 500

            return jsonify({
                "success": True,
                "analysis": {
                    "solution": mark_result['solution'],
                    "score": mark_result['score'],
                    "response": mark_result['response']
                }
            })
    except Exception as e:
        print(f"🔥 Unexpected error in analyze_document: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "Internal server error occurred",
            "details": str(e),
            "type": "unexpected_error"
        }), 500


# Global error handler for all unhandled exceptions
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"🚨 Unhandled exception: {str(e)}")
    traceback.print_exc()
    return jsonify({
        "success": False,
        "error": "An unexpected error occurred",
        "details": str(e),
        "type": "unhandled_exception"
    }), 500


# Error handler for 404 errors
@app.errorhandler(404)
def handle_404(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "details": f"The requested URL {request.url} was not found on this server",
        "available_endpoints": ["/", "/api/generate-stat-graph", "/api/analyze-document"]
    }), 404


# Error handler for 405 Method Not Allowed
@app.errorhandler(405)
def handle_405(e):
    return jsonify({
        "success": False,
        "error": "Method not allowed",
        "details": f"The method {request.method} is not allowed for {request.url}",
        "allowed_methods": ["POST"]
    }), 405


if __name__ == "__main__":
    debug_mode = current_environment == 'development'
    app.run(host="127.0.0.1", port=8000, debug=debug_mode)
