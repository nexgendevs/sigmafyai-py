# Third-party imports
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pandas as pd

# Local imports
import utils

import os

# Load environment variables
load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"message": "AI Sigmafy running updated..."})


@app.route('/api/generate-stat-graph', methods=['POST'])
def generate_stat_graph():
    try:
        if 'csv_url' not in request.form or 'graph_type' not in request.form:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        csv_url = request.form['csv_url']
        graph_type = request.form['graph_type'].lower()
        
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

        # Use robust CSV loading
        df = utils.load_csv_robust(csv_url)

        python_code = utils.generate_graph_with_llm(df, graph_type)

        encoded_img, error = utils.execute_python_code(python_code)

        if error:
            return jsonify({"success": False, "error": error})

        print(f"Generated code:\n\n{python_code}")
        return jsonify({"success": True, "base64_image": encoded_img})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    try:
        # Check for required fields
        required_fields = ['question_title', 'question_content', 'approval_criteria', 'txt_only_submission']
        for field in required_fields:
            if field not in request.form:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400

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
                return jsonify({"success": False, "error": "Internal error occurred while analyzing document"}), 500

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

                python_code = utils.generate_graph_with_llm(df, chart_type)
                base64_image, chart_error = utils.execute_python_code(python_code)

                if chart_error:
                    return jsonify({"success": False, "error": chart_error}), 500

                return jsonify({
                    "success": True,
                    "base64_image": base64_image
                })

            except Exception as chart_ex:
                return jsonify({"success": False, "error": str(chart_ex)}), 500

        # Handle text-only submission (fallback when no optional fields provided)
        else:
            print(f"Fallback to text-only submission...")
            # Handle text-only submission using mark_answer function
            mark_result = utils.mark_answer(question_title, question_content, approval_criteria, txt_only_submission)

            if not mark_result['success']:
                return jsonify(mark_result), 500

            return jsonify({
                "success": True,
                "analysis": {
                    "solution": mark_result['solution'],
                    "score": mark_result['score'],
                    "response": mark_result['response']
                }
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
