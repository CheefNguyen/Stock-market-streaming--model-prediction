document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('codeSelect').addEventListener('change', fetchRealtimeStockData);
    document.getElementById('calculateBtn').addEventListener('click', DQNAgentPredict);
});

function fetchRealtimeStockData() {
    const codeSelect = document.getElementById('codeSelect');
    const stockCode = codeSelect.value;
    fetch(`/get_realtime_stock_data?code=${stockCode}`)
        .then(response => response.json())
        .then(data => {
            data.sort((a, b) => new Date(a.date) - new Date(b.date));
            renderRealtimeChart(data);
        });
}

function DQNAgentPredict(){
    const codeSelect = document.getElementById('codeSelect');
    const stockCode = codeSelect.value;
    fetch(`/get_action_data?code=${stockCode}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            renderChart(data.dates, data.actions);
        })
        .catch(error => console.error('Error fetching action data:', error));
}

let stockRealtimeChart;
let actionChart ;

function renderRealtimeChart(data) {
    const labels = data.map(item => item.date);
    const adClose = data.map(item => item.adClose);

    const ctx = document.getElementById('realtimeChart').getContext('2d');

    if (stockRealtimeChart) {
        stockRealtimeChart.destroy();
    }

    stockRealtimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Adjusted Close Price',
                data: adClose,
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

function renderChart(dates, actions) {
    const ctx = document.getElementById('actionChart').getContext('2d');

    if (actionChart) {
        actionChart.destroy();
    }

    actionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: dates,
            datasets: [{
                label: 'Stock Actions',
                data: actions,
                backgroundColor: actions.map(action => {
                    if (action === 0) return 'blue'; // Hold
                    if (action === 1) return 'red'; // Sell
                    return 'green'; // Buy
                })
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
                        text: 'Action',
                        color: 'white'
                    },
                    ticks: {
                        color: 'white',
                        stepSize: 1,
                        callback: function(value) {
                            if (value === 0) return 'Hold';
                            if (value === 1) return 'Sell';
                            return 'Buy';
                        }
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