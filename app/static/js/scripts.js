document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('codeSelect').addEventListener('change', fetchRealtimeStockData);
    document.getElementById('calculateBtn').addEventListener('click', drawActionChart);
    
});

function drawActionChart() {
    const stockCode = document.getElementById('codeSelect').value;
    const initBalance = document.getElementById('balanceInput').value;
    fetch(`/get_action_data?code=${stockCode}&balance=${initBalance}`)
        .then(response => response.json())
        .then(data => {
            var dates = data.dates;
            var prices = data.prices;
            var actions = data.actions;
            var portfolio = data.portfolio;
            var balance = data.balance;

            // var scaledBalance = balance.map(b => b * 0.01);

            console.log(data);

            var trace1 = {
                x: dates,
                y: prices,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Stock Price'
            };

            var trace3 = {
                x: dates,
                y: balance,
                type: 'scatter',
                mode: 'lines',
                name: 'Balance',
                yaxis: 'y2'
            };

            var traces = [trace1, trace3];

            var buyAdded = false;
            var sellAdded = false;
            
            actions.forEach((action, index) => {
                if (action !== 0) {  // Skip 'hold' actions
                    var trace = {
                        x: [dates[index]],
                        y: [prices[index]],
                        mode: 'markers',
                        marker: {
                            size: 12,
                            symbol: action === 1 ? 'triangle-up' : 'triangle-down', // 'sell' or 'buy'
                            color: action === 1 ? 'green' : 'red' // Green for sell, red for buy
                        },
                        name: action === 1 ? 'Sell' : 'Buy',
                        showlegend: false
                    };
                    
                    // Add legend for the first occurrence of each action
                    if (action === 1 && !sellAdded) {
                        trace.showlegend = true;
                        sellAdded = true;
                    } else if (action === 2 && !buyAdded) {
                        trace.showlegend = true;
                        buyAdded = true;
                    }

                    traces.push(trace);
                }
            });

            // Add a text annotation for final profit
            var annotation = {
                x: dates[dates.length - 1],
                y: prices[prices.length - 1],
                xref: 'x',
                yref: 'y',
                text: `Portfolio: ${portfolio.toFixed(2)}`,
                arrowhead: 1,
                ax: 0,
                ay: 20
            };

            var layout = {
                title: 'Stock Price, Actions over time',
                xaxis: { title: 'Date', automargin: true },
                yaxis: { title: 'Price', automargin: true },
                yaxis2: {
                    overlaying: 'y',
                    side: 'right'
                },
                margin: { l: 50, r: 50, t: 50, b: 50 },  // Adjust margins to ensure sufficient space
                responsive: true,  // Make the plot responsive
                annotations: [annotation]  // Include the annotation
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