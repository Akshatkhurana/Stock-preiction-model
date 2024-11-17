import React, { useState } from 'react';
import axios from 'axios';
import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip } from 'recharts';

const App = () => {
    const [stockName, setStockName] = useState('');
    const [days, setDays] = useState(7);
    const [data, setData] = useState([]);

    const predictStock = () => {
        axios.post('http://localhost:5000/predict', { stockName, days })
            .then((res) => setData(res.data.predictions))
            .catch((err) => console.error(err));
    };

    return (
        <div>
            <input type="text" placeholder="Stock Name" value={stockName} onChange={(e) => setStockName(e.target.value)} />
            <input type="number" value={days} onChange={(e) => setDays(e.target.value)} />
            <button onClick={predictStock}>Predict</button>
            <LineChart width={600} height={300} data={data}>
                <Line type="monotone" dataKey="value" stroke="#8884d8" />
                <CartesianGrid stroke="#ccc" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
            </LineChart>
        </div>
    );
};

export default App;
