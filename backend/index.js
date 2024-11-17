const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(cors({ origin: 'http://localhost:5173' })); // Replace with your frontend URL
app.use(bodyParser.json());
app.get('/test', (req, res) => {
    res.json({ message: "Server is working!" });
});

app.post('/predict', (req, res) => {
    const { stockName, days } = req.body;

    console.log("Received request:", stockName, days);

    const python = spawn('python', ['predict.py', stockName, days]);

    let pythonOutput = "";
    let pythonError = "";

    // Capture standard output
    python.stdout.on('data', (data) => {
        console.log("Raw Python Output:", data.toString());
        pythonOutput += data.toString();
    });

    // Capture standard error
    python.stderr.on('data', (data) => {
        pythonError += data.toString();
        console.error("Python Error:", data.toString());
    });
    
    // Handle process completion
    python.on('close', (code) => {
        if (code === 0) {
            try {
                const predictions = JSON.parse(pythonOutput.trim());
                res.json({ predictions: predictions.map((value, index) => ({ date: `Day ${index + 1}`, value })) });
            } catch (err) {
                console.error("Error parsing Python output:", err);
                res.status(500).send("Invalid output format from Python script.");
            }
        } else {
            res.status(500).send(`Python script failed: ${pythonError}`);
        }
    });
    
});

app.listen(5000, () => console.log('Server running on port 5000'));