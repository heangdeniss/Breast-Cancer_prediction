import express from "express";
import bodyParser from "body-parser";

const app = express();
const port = 5000;

app.use(express.static('public'));
app.use(bodyParser.urlencoded({ extended: true}));



// Logistic regression model parameters
const intercept = -0.9050;
const theta_values = [
    1.0101,   // radius_mean
    0.3257,   // texture_mean
    0.9703,   // perimeter_mean
    -0.3533,  // symmetry_mean
    1.6282,   // radius_se
    -0.1226,  // concave_points_se
    1.5353,   // radius_worst
    1.2423,   // texture_worst
    0.9967,   // smoothness_worst
    1.7998,   // concave_points_worst
    0.9784,   // symmetry_worst
    -0.2384   // fractal_dimension_worst
];
const mean_values = [
    14.127, 19.289, 91.969, 0.181, 0.278,
    0.028,  16.269, 25.677, 0.132, 0.254,
    0.290,  0.083
];
const std_values = [
    3.521, 4.301, 24.298, 0.027, 0.197,
    0.010, 4.833, 7.336, 0.022, 0.071,
    0.061, 0.018
];

// render welcome page
app.get("/", (req, res) => {
    res.render("homepage.ejs");
});
// display feature 
app.get("/feature", (req, res) => {
    res.render("feature.ejs")
});

// Route for predict page.
app.get('/index', (req, res) => {
    res.render('index.ejs');
});

// Handle prediction form submission
app.post('/predict', (req, res) => {
    const input_features = [
        parseFloat(req.body.radius_mean),
        parseFloat(req.body.texture_mean),
        parseFloat(req.body.perimeter_mean),
        parseFloat(req.body.symmetry_mean),
        parseFloat(req.body.radius_se),
        parseFloat(req.body.concave_points_se),
        parseFloat(req.body.radius_worst),
        parseFloat(req.body.texture_worst),
        parseFloat(req.body.smoothness_worst),
        parseFloat(req.body.concave_points_worst),
        parseFloat(req.body.symmetry_worst),
        parseFloat(req.body.fractal_dimension_worst)
    ];

    const scaled_features = input_features.map((feature, index) => (feature - mean_values[index]) / std_values[index]);

    let logit_p = intercept;
    for (let i = 0; i < scaled_features.length; i++) {
        logit_p += scaled_features[i] * theta_values[i];
    }
    const probability = 1 / (1 + Math.exp(-logit_p));
    const diagnosis = probability >= 0.5 ? 'Malignant' : 'Benign';

    res.render('result.ejs', {
        logit_p: logit_p.toFixed(4),
        probability: (probability * 100).toFixed(2),
        diagnosis: diagnosis
    });
});






app.listen(port, () => {
    console.log(`Chill guys the sever is running on ${port}`);
});