from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # Generate X values from a uniform distribution
    x = np.random.uniform(low=-20, high=20, size=N)
    
    # Introduce more variability to Y by adding more noise and a non-linear component
    y = 2 * x + np.random.normal(0, np.sqrt(4 * sigma2), N) + np.random.uniform(-10, 10, N)

    # Fit linear regression using numpy's polyfit
    slope, intercept = np.polyfit(x, y, 1)
    
    # Generate plot1: regression plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data points', alpha=0.5, color='purple')
    plt.plot(x, slope * x + intercept, color='red', label='Fitted line')
    plt.title('Linear Regression with Random Data')

    # Add the linear fit equation to the plot
    equation_text = f'Y = {slope:.2f}X + {intercept:.2f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plot1_path = 'static/regression_plot.png'
    plt.savefig(plot1_path)
    plt.close()

    # Generate histograms for slopes and intercepts from multiple simulations
    slopes = []
    intercepts = []

    for _ in range(S):
        x_sim = np.random.uniform(low=-20, high=20, size=N)
        y_sim = 2 * x_sim + np.random.normal(0, np.sqrt(4 * sigma2), N) + np.random.uniform(-10, 10, N)
        slope_sim, intercept_sim = np.polyfit(x_sim, y_sim, 1)
        slopes.append(slope_sim)
        intercepts.append(intercept_sim)

    # Generate plot2: histogram of slopes and intercepts
    plt.figure(figsize=(8, 6))
    
    # Histogram for slopes
    plt.hist(slopes, bins=30, alpha=0.5, color='blue', label='Slopes')
    plt.axvline(x=slope, color='red', linestyle='dashed', linewidth=2, label='Observed slope')

    # Histogram for intercepts
    plt.hist(intercepts, bins=30, alpha=0.5, color='green', label='Intercepts')
    plt.axvline(x=intercept, color='orange', linestyle='dashed', linewidth=2, label='Observed intercept')

    plt.title('Histogram of Slopes and Intercepts')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plot2_path = 'static/histogram_plot.png'
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of extreme slopes and intercepts
    slope_extreme = np.mean(np.abs(np.array(slopes) - slope) > np.abs(slope))
    intercept_extreme = np.mean(np.abs(np.array(intercepts) - intercept) > np.abs(intercept))

    return plot1_path, plot2_path, slope_extreme, intercept_extreme


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)