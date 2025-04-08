document.addEventListener('DOMContentLoaded', () => {
    const stockSymbolInput = document.getElementById('stockSymbol');
    const predictBtn = document.getElementById('predictBtn');
    const predictionResults = document.getElementById('predictionResults');
    const symbolElement = document.getElementById('symbol');
    const currentPriceElement = document.getElementById('currentPrice');
    const predictedPriceElement = document.getElementById('predictedPrice');
    const explanationElement = document.getElementById('explanation');
    const notificationToast = document.getElementById('notificationToast');
    const toast = new bootstrap.Toast(notificationToast);
    let featureImportanceChart = null;
    let tradingViewWidget = null; // Variable to hold the TradingView widget

    predictBtn.addEventListener('click', async () => {
        const priceChartData = {
            labels: [],
            datasets: [{
                label: 'Stock Price',
                data: [],
                borderColor: 'rgba(30, 60, 114, 1)',
                backgroundColor: 'rgba(30, 60, 114, 0.6)',
                fill: false
            }]
        };

        const symbol = stockSymbolInput.value.trim().toUpperCase();
        
        if (!symbol) {
            showNotification('Please enter a stock symbol', 'error');
            return;
        }

        try {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';

            // Show spinner
            loadingSpinner.style.display = 'block';
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbol }),
            });

            const data = await response.json();

            if (response.ok) {
                // Update prediction results
                symbolElement.textContent = `Symbol: ${data.symbol}`;
                currentPriceElement.textContent = `Current Price: $${data.current_price}`;
                predictedPriceElement.textContent = `Predicted Price: $${data.predicted_price}`;
                
                // Update explanation
                explanationElement.innerHTML = generateExplanationHTML(data);
                document.getElementById('buyPrice').textContent = `Buy Price: $${data.buy_price}`;
                document.getElementById('sellPrice').textContent = `Sell Price: $${data.sell_price}`;
                document.getElementById('buySellExplanation').textContent = data.buy_sell_explanation;

                // Update feature importance chart
                updateFeatureImportanceChart(data.explanation.feature_importance);
                
                // Update price chart
                const historicalData = data.historical_data;
                const dates = data.dates;
                priceChartData.labels = dates;
                priceChartData.datasets[0].data = historicalData;
                renderPriceChart(priceChartData);
                
                // Show results
                predictionResults.classList.remove('d-none');
                predictionResults.classList.add('show');
                
                // Update TradingView chart with the new symbol
                if (tradingViewWidget) {
                    tradingViewWidget.remove(); // Remove the existing widget
                }
                tradingViewWidget = new TradingView.widget({
                    "container_id": "tradingview-chart",
                    "width": "100%",
                    "height": "400",
                    "symbol": symbol, // Use the user input symbol
                    "interval": "D",
                    "timezone": "Etc/UTC",
                    "theme": "light",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "allow_symbol_change": true,
                    "hideideas": true,
                    "studies": ["MACD@tv-basicstudies"]
                });

                showNotification('Prediction completed successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to get prediction');
            }
        } catch (error) {
            showNotification(error.message, 'error');
        } finally {
            // Hide spinner
            loadingSpinner.style.display = 'none';
            predictBtn.disabled = false;
            predictBtn.textContent = 'Predict Price';
        }
    });

    async function fetchStockOfTheDay() {
        try {
            const response = await fetch('/stock-of-the-day');
            const data = await response.json();
            if (response.ok) {
                document.getElementById('stockOfTheDay').textContent = `Stock of the Day: ${data.stock_of_the_day} (Predicted Profit: ${data.predicted_profit}%)`;
            } else {
                throw new Error(data.error || 'Failed to fetch stock of the day');
            }
        } catch (error) {
            console.error('Error fetching stock of the day:', error);
        }
    }

    function generateExplanationHTML(data) {
        fetchStockOfTheDay();

        const priceChange = data.price_change > 0 ? 'increase' : 'decrease';
        
        // Generate main explanation
        let html = `
            <div class="explanation-text">
            <p>Based on our analysis, the stock price of ${data.symbol} is predicted to ${priceChange} 
            by $${Math.abs(data.price_change).toFixed(2)} (${data.price_change_percent.toFixed(2)}%).</p>
            <p>The suggested buy price of $${data.buy_price} is calculated as 2% below the predicted price, while the suggested sell price of $${data.sell_price} is 2% above the predicted price. These values are influenced by various factors including market trends, historical performance, and technical indicators.</p>
            <p class="text-muted mb-3">Last updated: ${data.last_updated}</p>
            </div>`;
        
        // Generate detailed explanations accordion
        html += `<div class="accordion" id="explanationAccordion">`;
        data.explanation.detailed_explanations.forEach((exp, index) => {
            html += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading${index}">
                        <button class="accordion-button ${index !== 0 ? 'collapsed' : ''}" 
                                type="button" data-bs-toggle="collapse" 
                                data-bs-target="#collapse${index}" 
                                aria-expanded="${index === 0 ? 'true' : 'false'}" 
                                aria-controls="collapse${index}">
                            ${exp.feature} (${(exp.importance * 100).toFixed(2)}% influence)
                        </button>
                    </h2>
                    <div id="collapse${index}" 
                         class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                         aria-labelledby="heading${index}" 
                         data-bs-parent="#explanationAccordion">
                        <div class="accordion-body">
                            <p>${exp.description}</p>
                            <p>Impact: ${exp.impact.toFixed(4)}</p>
                        </div>
                    </div>
                </div>`;
        });
        html += `</div>`;
        
        // Generate overall impact cards
        html += `
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card bg-success bg-opacity-10">
                        <div class="card-body">
                            <h6>Positive Factors</h6>
                            <p>Total positive impact: ${data.explanation.overall_impact.positive.toFixed(4)}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card bg-danger bg-opacity-10">
                        <div class="card-body">
                            <h6>Negative Factors</h6>
                            <p>Total negative impact: ${data.explanation.overall_impact.negative.toFixed(4)}</p>
                        </div>
                    </div>
                </div>
            </div>`;
        
        return html;
    }

    function updateFeatureImportanceChart(featureImportance) {
        const ctx = document.getElementById('featureImportanceChart').getContext('2d');
        const labels = Object.keys(featureImportance);
        const data = Object.values(featureImportance);

        // Destroy existing chart if it exists
        if (featureImportanceChart) {
            featureImportanceChart.destroy();
        }

        featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: data,
                    backgroundColor: 'rgba(30, 60, 114, 0.6)',
                    borderColor: 'rgba(30, 60, 114, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance Score'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Features'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Importance: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

async function fetchLiveStockData(symbol) {
    const response = await fetch(`https://api.example.com/live?symbol=${symbol}`);

    if (!response.ok) {
        throw new Error('Failed to fetch live stock data');
    }
    return await response.json();
}

function renderPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
}

function showNotification(message, type) {
    const toastBody = notificationToast.querySelector('.toast-body');
    toastBody.textContent = message;
    notificationToast.classList.remove('bg-success', 'bg-danger');
    notificationToast.classList.add(type === 'success' ? 'bg-success' : 'bg-danger');
    toast.show();
}
});
