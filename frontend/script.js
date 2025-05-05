fetch("https://shl-recommendation-engine-3bbj.onrender.com/recommendations", {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({ jd_text: jdText })
})
.then(async response => {
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Something went wrong");
    }
    return response.json();
})
.then(data => {
    recommendationsDiv.innerHTML = "";

    if (data.recommendations && data.recommendations.length > 0) {
        let tableHTML = "<table><tr><th>Assessment Name</th><th>Duration</th><th>Type</th></tr>";
        data.recommendations.forEach(rec => {
            tableHTML += `<tr>
                <td>${rec.name}</td>
                <td>${rec.duration_min} mins</td>
                <td>${rec.type}</td>
            </tr>`;
        });
        tableHTML += "</table>";
        recommendationsDiv.innerHTML = tableHTML;
    } else {
        recommendationsDiv.innerHTML = "<p>No relevant assessments found.</p>";
    }
})
.catch(error => {
    console.error("Error:", error);
    recommendationsDiv.innerHTML = `<p style="color:red;">Error: ${error.message}</p>`;
});
