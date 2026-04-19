"use strict";

const API_URL = "http://127.0.0.1:8000/api/v1/predict";

// ---------- GET FORM DATA ----------
function getFormData() {
    return {
        Pregnancies: Number(document.getElementById("pregnancies").value),
        Glucose: Number(document.getElementById("glucose").value),
        BloodPressure: Number(document.getElementById("bloodPressure").value),
        SkinThickness: Number(document.getElementById("skinThickness").value),
        Insulin: Number(document.getElementById("insulin").value),
        BMI: Number(document.getElementById("bmi").value),
        DiabetesPedigreeFunction: Number(document.getElementById("dpf").value),
        Age: Number(document.getElementById("age").value)
    };
}

// ---------- API CALL ----------
async function predictDiabetes(data) {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        return await response.json();

    } catch (error) {
        console.error(error);
        alert("❌ Backend connect nahi ho raha");
    }
}

// ---------- SHOW RESULT ----------
function showResult(result) {
    const box = document.getElementById("resultBox");

    box.style.display = "block";

    box.innerHTML = `
        <h2>${result.label}</h2>
        <p><b>Confidence:</b> ${result.confidence}</p>
        <p><b>Risk Level:</b> ${result.risk_level.label}</p>
        <p><b>Probability:</b> ${(result.probability_diabetic * 100).toFixed(2)}%</p>
    `;
}

// ---------- BUTTON CLICK ----------
document.getElementById("predictBtn").addEventListener("click", async () => {
    const data = getFormData();

    const result = await predictDiabetes(data);

    if (result) {
        showResult(result);
    }
});
document.querySelector("form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const data = getFormData();
    const result = await predictDiabetes(data);

    if (result) {
        showResult(result);
    }
});
document.getElementById("predictBtn").addEventListener("click", async (e) => {
    e.preventDefault();

    const btn = document.getElementById("predictBtn");

    // 🔥 LOADING START
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    const data = getFormData();
    const result = await predictDiabetes(data);

    // 🔥 LOADING END
    btn.innerText = "Analyse Risk Profile";
    btn.disabled = false;

    if (result) {
        showResult(result);
    }
});