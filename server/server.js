const express = require("express");
const path = require("path");

const app = express();
const PORT = 3001;

// Serve static files from public folder
app.use(express.static(path.join(__dirname, "public")));

// Parse JSON bodies
app.use(express.json());

const { spawn } = require("child_process");

app.post("/admin/predict", (req, res) => {
  const inputData = req.body;
  console.log("Received:", inputData);

  const py = spawn("python", ["predict.py"]);
  let result = "";

  py.stdin.write(JSON.stringify(inputData));
  py.stdin.end();

  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
  });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: "Prediction failed." });
    }

    try {
      const parsed = JSON.parse(result);
      res.json(parsed);
    } catch (e) {
      console.error("JSON parse error:", e);
      res.status(500).json({ error: "Invalid response from model." });
    }
  });
});

// Optional: fallback route to serve index.html for any unmatched route
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});

