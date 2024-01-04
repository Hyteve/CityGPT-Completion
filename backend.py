from flask import Flask, request, jsonify, render_template, send_file
import sys
import os
from generate.generate import main
import cv2
import base64
import logging
import json

app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    try:
        status = 200
        logging.info("Status: 200")
    except Exception as e:
        print(e)
        status = 400
    return jsonify(response='', status=status, mimetype='application/json')

@app.route("/invocations",methods=["POST"])
def invoke():
    try:
        # json_data = request.json()
        # data = json_data.get('scence')
        # box_start_point = [json_data['x'], json_data['y']]
        # img = main(json_data, box_start_point)
        # cv2.imwrite('generate.jpg',img)

        print("test")
        json_path = './004.json'
        box_start_point = [800, 300]
        
        with open(json_path, 'r') as infile:
            json_data = json.load(infile)
            
        data = json_data.get('scence')
        img = main(data, box_start_point)
        cv2.imwrite('generate.jpg',img)


    except Exception as e:
        print("Error: ", e)

    return send_file('generate.jpg', mimetype='image/jpeg')

@app.route("/test",methods=["GET"])
def test():
    try:
        print("test")
        json_path = './004.json'
        box_start_point = [800, 300]
        
        with open(json_path, 'r') as infile:
            json_data = json.load(infile)
            
        data = json_data.get('scence')
        img = main(json_data, box_start_point)
        cv2.imwrite('generate.jpg',img)


    except Exception as e:
        print("Error: ", e)

    return send_file('generate.jpg', mimetype='image/jpeg')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0', port=port)