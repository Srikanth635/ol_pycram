from flask import Flask, request, jsonify
from .src.sv_graph import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/update' , methods=['POST'])
def update_designator():
    try:
        # Get data from request (works with JSON or form-data)
        data = request.get_json() if request.is_json else request.form

        # Extract parameters
        _instruction = data.get('instruction')
        _action_designator = data.get('action_designator')
        _reason_for_failure = data.get('reason_for_failure',"")
        _human_comment = data.get('human_comment',"")

        # Validate required fields
        # if not _action_designator:
        #     return jsonify({'error': 'action_designator is required'}), 400

        # Process the data (here we just echo it back)
        # parsed_response = {
        #     'status': 'success',
        #     'data': {
        #         'action_designator': action_designator,
        #         'reason_for_failure': reason_for_failure,
        #         'human_comment': human_comment
        #     }
        # }

        # Model Invocation
        _config = {"configurable": {"thread_id": 1}}

        if not _instruction:
            if not _action_designator:
                return jsonify({'error': 'action_designator/instruction is required'}), 400
            else:
                final_graph_state = sv_grapher.invoke(
                    {"action_designator": _action_designator, "reason_for_failure": _reason_for_failure,
                     "human_comment": _human_comment}, config=_config, stream_mode="updates")
                model_failure_reasoning = sv_grapher.get_state(_config).values["failure_reasons_solutions"]
                parameters_updated = sv_grapher.get_state(_config).values["update_parameters_reasons"]
                updated_action_designator = sv_grapher.get_state(_config).values["updated_action_designator"]

                human_instruction_dict = sv_grapher.get_state(config=_config).values['human_instruction']
                human_instruction = human_instruction_dict.get('ad_instruction', "")

                model_response = {
                    'updated_action_designator': str(updated_action_designator),
                    'model_failure_reasoning': model_failure_reasoning,
                    'parameters_updated': parameters_updated,
                    'human_instruction': human_instruction
                }
        else:
            final_graph_state = sv_grapher.invoke({"instruction" : _instruction}, config = _config,stream_mode="updates")
            updated_action_designator = sv_grapher.get_state(_config).values["updated_action_designator"]
            model_response = {
                'updated_action_designator': str(updated_action_designator)
            }


        return jsonify(model_response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)