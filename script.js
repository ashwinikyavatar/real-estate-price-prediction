function predictPrice() {

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
.then(response => response.json())
.then(data => {

    const formattedPrice = Number(data.predicted_price).toLocaleString(
    "en-AU",
    {
        style: "currency",
        currency: "AUD",
        maximumFractionDigits: 0
    }
);

    document.getElementById("price").innerText = formattedPrice;

    document.getElementById("message").innerText =
        "Prediction generated using the trained Machine Learning model.";
})
.catch(error => {
    console.error(error);

    alert("Unable to connect to the prediction server. Please make sure Flask is running.");
});
}