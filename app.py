from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats

app = Flask(__name__)

# Function to generate plots based on user input
def generate_plots(N, mu, sigma2, S, beta0, beta1):
    # Generate X values (independent variable)
    X = np.random.normal(mu, np.sqrt(sigma2), N)
    # Generate Y values (dependent variable)
    Y = beta0 + beta1 * X + np.random.normal(0, np.sqrt(sigma2), N)

    # Plot the data and the true line
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, Y, label='Data', color='blue')
    ax1.plot(X, beta0 + beta1 * X, label='True Line', color='red')
    ax1.set_title('Data and True Line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    # Simulate multiple slopes from the data
    slopes = []
    for _ in range(S):
        Y_sample = beta0 + beta1 * X + np.random.normal(0, np.sqrt(sigma2), N)
        slope, intercept, _, _, _ = stats.linregress(X, Y_sample)
        slopes.append(slope)

    # Plot histogram of simulated slopes
    fig2, ax2 = plt.subplots()
    ax2.hist(slopes, bins=30, color='green', alpha=0.7)
    ax2.axvline(x=beta1, color='red', linestyle='--', label='True Slope')
    ax2.set_title('Histogram of Simulated Slopes')
    ax2.set_xlabel('Slope')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Encode the plots as base64 strings
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return plot1_base64, plot2_base64, slopes

# Route to handle the home page and data generation
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        try:
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            S = int(request.form["S"])
            beta0 = float(request.form["beta0"])
            beta1 = float(request.form["beta1"])

            # Generate plots and simulations
            plot1, plot2, slopes = generate_plots(N, mu, sigma2, S, beta0, beta1)
            return render_template("index.html", plot1=plot1, plot2=plot2, slopes=slopes)

        except Exception as e:
            return render_template("index.html", error="Error generating data: " + str(e))

    return render_template("index.html")

# Route for handling Hypothesis Testing
@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    try:
        test_param = request.form["test_param"]
        test_type = request.form["test_type"]
        observed_stat = float(request.form["observed_stat"])

        # Load the slopes from the previous simulation (you would typically want to store them globally)
        slopes = request.args.get("slopes", default=None)  # Retrieve the slopes from the data generation

        if not slopes:
            return redirect(url_for('index'))

        # Hypothesis Testing Logic
        p_value = None
        if test_param == "slope":
            # Perform hypothesis test for the slope
            if test_type == ">":
                p_value = (np.array(slopes) >= observed_stat).mean()
            elif test_type == "<":
                p_value = (np.array(slopes) <= observed_stat).mean()
            elif test_type == "!=":
                p_value = (np.abs(np.array(slopes) - observed_stat) >= observed_stat).mean()

        elif test_param == "intercept":
            # Similar logic would apply for intercept testing if you added intercept calculation
            pass

        message = f"P-value: {p_value:.4f}"
        if p_value <= 0.0001:
            message += " - This is a rare event!"
        
        return render_template("index.html", p_value=p_value, message=message)

    except Exception as e:
        return render_template("index.html", error="Error during hypothesis testing: " + str(e))

# Route for handling Confidence Interval Calculation
@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        ci_param = request.form["ci_param"]
        confidence_level = float(request.form["confidence_level"])

        # Load slopes from the data generation (you would typically want to store them globally)
        slopes = request.args.get("slopes", default=None)  # Retrieve the slopes

        if not slopes:
            return redirect(url_for('index'))

        # Confidence Interval Calculation Logic
        mean_estimate = np.mean(slopes)
        std_error = np.std(slopes) / np.sqrt(len(slopes))

        # Calculate the confidence interval using the t-distribution
        ci = stats.t.interval(confidence_level, len(slopes) - 1, loc=mean_estimate, scale=std_error)

        return render_template("index.html", confidence_interval=ci, mean_estimate=mean_estimate)

    except Exception as e:
        return render_template("index.html", error="Error calculating confidence interval: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
