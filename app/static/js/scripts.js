document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('codeSelect').addEventListener('change', fetchRealtimeStockData);
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

let stockRealtimeChart;
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