from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

data = None
kmeans = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initialize', methods=['POST'])
def initialize():
    global data, kmeans
    data = np.array(request.json['data'])
    method = request.json['method']
    k = int(request.json['k'])
    kmeans = KMeans(k=k)
    kmeans.initialize_centroids(data, method)
    return jsonify({'centroids': kmeans.centroids.tolist()})

@app.route('/step', methods=['POST'])
def step():
    global kmeans, data
    kmeans.fit(data)
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'labels': kmeans.labels.tolist()
    })

@app.route('/run', methods=['POST'])
def run():
    global kmeans, data
    kmeans.fit(data)
    return jsonify({
        'centroids': kmeans.centroids.tolist(),
        'labels': kmeans.labels.tolist()
    })

@app.route('/reset', methods=['POST'])
def reset():
    global kmeans, data
    data = None
    kmeans = None
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
