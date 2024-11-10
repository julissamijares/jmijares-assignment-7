from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Necessary for session management

# Function to generate plots based on user input
def generate_plots(N, mu, sigma2, S, beta0, beta1):
    X = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + np.random.normal(0, np.sqrt(sigma2), N)

    # Plot the data and the true line
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, Y, label='Data', color='blue')
    ax1.plot(X, beta0 + beta1 * X, label='True Line', color='red')
    ax1.set_title('Data and True Line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    slopes = []
    for _ in range(S):
        Y_sample = beta0 + beta1 * X + np.random.normal(0, np.sqrt(sigma2), N)
        slope, _, _, _, _ = stats.linregress(X, Y_sample)
        slopes.append(slope)

    # Plot histogram of simulated slopes
    fig2, ax2 = plt.subplots()
    ax2.hist(slopes, bins=30, color='green', alpha=0.7)
    ax2.axvline(x=beta1, color='red', linestyle='--', label='True Slope')
    ax2.set_title('Histogram of Simulated Slopes')
    ax2.set_xlabel('Slope')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return plot1_base64, plot2_base64, slopes

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

            plot1, plot2, slopes = generate_plots(N, mu, sigma2, S, beta0, beta1)
            session['slopes'] = slopes  # Store slopes in session for other routes to access
            return render_template("index.html", plot1=plot1, plot2=plot2)

        except Exception as e:
            return render_template("index.html", error="Error generating data: " + str(e))

    return render_template("index.html")

# Hypothesis Test Plot
def hypothesis_test_visualization(slopes, observed_stat, hypothesized_stat):
    fig, ax = plt.subplots()
    ax.hist(slopes, bins=30, color='blue', alpha=0.7)
    
    # Red dashed line for the observed slope (from the sample)
    ax.axvline(np.median(slopes), color='red', linestyle='--', linewidth=2, label="Observed Slope")
    # Blue line for the hypothesized slope
    ax.axvline(hypothesized_stat, color='blue', linestyle='-', linewidth=2, label="Hypothesized Slope (H₀)")  
    ax.set_title("Hypothesis Testing: Simulated Slopes with Observed & Hypothesized Values")
    ax.set_xlabel("Slope")
    ax.set_ylabel("Frequency")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64


# Confidence Interval Visualization
def confidence_interval_visualization(slopes, ci, true_param, mean_estimate):
    fig, ax = plt.subplots()
    ax.scatter(slopes, [1] * len(slopes), color='gray', alpha=0.6, label="Simulated Estimates")
    # Blue horizontal line spanning the confidence interval
    ax.hlines(1, ci[0], ci[1], color='blue', linewidth=6, label="Confidence Interval")
    ax.scatter(mean_estimate, 1, color='blue', s=100, label="Mean Estimate")
    ax.axvline(true_param, color='red', linestyle='--', linewidth=2, label="True Parameter")
    ax.set_yticks([])
    ax.set_title("Confidence Interval Visualization")
    ax.set_xlabel("Slope Value")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    try:
        # Get the form data
        test_param = request.form["test_param"]
        test_type = request.form["test_type"]

        # Retrieve slopes from session
        slopes = np.array(session.get("slopes", []))
        if len(slopes) == 0:
            return redirect(url_for('index'))

        # Determine the hypothesized value based on test_param
        if test_param == "slope":
            hypothesized_stat = session.get("beta1", 0)  # Get the true slope value
        elif test_param == "intercept":
            hypothesized_stat = session.get("beta0", 0)  # Get the true intercept value

        # Calculate the observed statistic (the sample median of slopes)
        observed_stat = np.median(slopes)  # Use the median as the observed statistic

        # Perform the hypothesis test
        if test_type == ">":
            p_value = (slopes >= observed_stat).mean()
        elif test_type == "<":
            p_value = (slopes <= observed_stat).mean()
        elif test_type == "!=":
            p_value = (np.abs(slopes - observed_stat) >= np.abs(hypothesized_stat - observed_stat)).mean()

        # Generate the hypothesis test plot
        test_plot = hypothesis_test_visualization(slopes, observed_stat, hypothesized_stat)

        return render_template("index.html", hypothesis_test_result=(p_value, f"P-value: {p_value:.4f}"), test_plot=test_plot)

    except Exception as e:
        return render_template("index.html", error="Error during hypothesis testing: " + str(e))

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        ci_param = request.form["ci_param"]
        confidence_level = float(request.form["confidence_level"])

        slopes = np.array(session.get("slopes", []))
        if len(slopes) == 0:
            return redirect(url_for('index'))

        mean_estimate = np.mean(slopes)
        std_error = np.std(slopes) / np.sqrt(len(slopes))
        ci = stats.t.interval(confidence_level, len(slopes) - 1, loc=mean_estimate, scale=std_error)
        true_param = session.get("beta1", 0)  # Retrieving beta1 for slope testing as true parameter

        ci_plot = confidence_interval_visualization(slopes, ci, true_param, mean_estimate)
        return render_template("index.html", confidence_interval_result=(ci, mean_estimate), ci_plot=ci_plot)

    except Exception as e:
        return render_template("index.html", error="Error calculating confidence interval: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
