window.predictPrice = function() {

    const livingArea =
        Number(document.getElementById("livingArea").value);

    const bedrooms =
        Number(document.getElementById("bedrooms").value);

    const bathrooms =
        Number(document.getElementById("bathrooms").value);

    const yearBuilt =
        Number(document.getElementById("yearBuilt").value);

    const garage =
        Number(document.getElementById("garage").value);

    const quality =
        Number(document.getElementById("quality").value);


    // Check all inputs
    if (
        !livingArea ||
        !bedrooms ||
        !bathrooms ||
        !yearBuilt ||
        !garage ||
        !quality
    ) {
        alert("Please enter all property details.");
        return;
    }


    // Send data to Flask API
    fetch("https://real-estate-price-prediction-505x.onrender.com/predict", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            living_area: livingArea,
            bedrooms: bedrooms,
            bathrooms: bathrooms,
            year_built: yearBuilt,
            garage: garage,
            quality: quality
        })

    })

    .then(response => {

        if (!response.ok) {
            throw new Error("Server returned HTTP " + response.status);
        }

        return response.json();

    })

    .then(data => {

        const formattedPrice =
            Number(data.predicted_price).toLocaleString(
                "en-AU",
                {
                    style: "currency",
                    currency: "AUD",
                    maximumFractionDigits: 0
                }
            );


        document.getElementById("price").innerText =
            formattedPrice;


        document.getElementById("message").innerText =
            "Prediction generated using the trained Machine Learning model.";

    })

    .catch(error => {

        console.error("Prediction error:", error);

        alert("Prediction failed: " + error.message);

    });

};