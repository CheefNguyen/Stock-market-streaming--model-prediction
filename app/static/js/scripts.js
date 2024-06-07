document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('codeSelect').addEventListener('change', fetchRealtimeStockData);
    document.getElementById('calculateBtn').addEventListener('click', drawActionChart);
    
});

function drawActionChart(){
    const stockCode = document.getElementById('codeSelect').value;
    fetch(`/get_action_data?code=${stockCode}`)
        .then(response => response.json())
        .then(data => {
            var dates = data.dates;
            var prices = data.prices;
            var actions = data.actions;

            console.log(data)

            var trace1 = {
                x: dates,
                y: prices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Stock Price'
            };

            var traces = [trace1];
            actions.forEach((action, index) => {
                if (action !== 0) {  // Skip 'hold' actions
                    traces.push({
                        x: [dates[index]],
                        y: [prices[index]],
                        mode: 'markers',
                        marker: {
                            size: 12,
                            symbol: action === 2 ? 'triangle-up' : 'triangle-down', // 'buy' or 'sell'
                            color: action === 2 ? 'green' : 'red' // Green for buy, red for sell
                        },
                        name: action === 1 ? 'Buy' : 'Sell'
                    });
                }
            });

            var layout = {
                title: 'Stock Price and Actions',
                xaxis: { title: 'Date', automargin: true },
                yaxis: { title: 'Price', automargin: true },
                margin: { l: 50, r: 50, t: 50, b: 50 },  // Adjust margins to ensure sufficient space
                responsive: true  // Make the plot responsive
            };

            Plotly.newPlot('actionChart', traces, layout);
        })
}

function fetchRealtimeStockData() {
    const stockCode = document.getElementById('codeSelect').value;
    fetch(`/get_realtime_stock_data?code=${stockCode}`)
        .then(response => response.json())
        .then(data => {
            data.sort((a, b) => new Date(a.date) - new Date(b.date));
            renderRealtimeChart(data);
        });
}

let stockRealtimeChart;

function renderRealtimeChart(data) {
    const labels = data.map(item => item.date);
    const close = data.map(item => item.close);

    const ctx = document.getElementById('realtimeChart').getContext('2d');

    if (stockRealtimeChart) {
        stockRealtimeChart.destroy();
    }

    stockRealtimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Close Price',
                data: close,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                fill: false,
                pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                pointBorderColor: 'rgba(75, 192, 192, 1)'
            }]
        },
        options: {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: 'white'
                    },
                    ticks: {
                        color: 'white'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price',
                        color: 'white'
                    },
                    ticks: {
                        color: 'white'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: 'white'
                    }
                }
            }
        }
    });
}