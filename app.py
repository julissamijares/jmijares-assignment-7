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
    fig1, ax1 = plt.subplots(figsize=(8, 6))  # Set a more reasonable size
    ax1.scatter(X, Y, label='Data', color='blue')
    ax1.plot(X, beta0 + beta1 * X, label='True Line', color='red')
    ax1.set_title('Data and True Line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()

    slopes = []
    intercepts = [] 
    for _ in range(S):
        Y_sample = beta0 + beta1 * X + np.random.normal(0, np.sqrt(sigma2), N)
        slope, intercept, _, _, _ = stats.linregress(X, Y_sample)
        slopes.append(slope)
        intercepts.append(intercept)

    # Plot histogram of simulated slopes and intercepts
    fig2, ax2 = plt.subplots(figsize=(8, 6)) 
    ax2.hist(slopes, bins=30, color='blue', alpha=0.7, label='Slope', density=True)
    ax2.axvline(x=beta1, color='blue', linestyle='--', label='True Slope')

    # Overlay histogram for intercepts
    ax2.hist(intercepts, bins=30, color='orange', alpha=0.7, label='Intercept', density=True)
    ax2.axvline(x=beta0, color='orange', linestyle='--', label='True Intercept')

    ax2.set_title('Histogram of Simulated Slopes and Intercepts')
    ax2.set_xlabel('Parameter Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    buf1.seek(0)
    plot1_base64 = base64.b64encode(buf1.getvalue()).decode('utf-8')

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')  # Use bbox_inches='tight' to remove unnecessary margins
    buf2.seek(0)
    plot2_base64 = base64.b64encode(buf2.getvalue()).decode('utf-8')

    # Store both slopes and intercepts in session
    session['slopes'] = slopes
    session['intercepts'] = intercepts  # Save intercepts for hypothesis testing
    session.modified = True  # Force session to commit

    return plot1_base64, plot2_base64, slopes, intercepts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Store values in the session after each form submission
            N = int(request.form["N"])
            mu = float(request.form["mu"])
            sigma2 = float(request.form["sigma2"])
            S = int(request.form["S"])
            beta0 = float(request.form["beta0"])
            beta1 = float(request.form["beta1"])

            # Save parameters in session for persistence
            session['N'] = N
            session['mu'] = mu
            session['sigma2'] = sigma2
            session['S'] = S
            session['beta0'] = beta0
            session['beta1'] = beta1

            plot1, plot2, slopes, intercepts = generate_plots(N, mu, sigma2, S, beta0, beta1)
            session['slopes'] = slopes  # Store slopes in session for other routes to access
            session["intercepts"] = intercepts  # Store intercepts in session
            print("Intercepts stored in session:", intercepts)
            return render_template("index.html", plot1=plot1, plot2=plot2, N=N, mu=mu, sigma2=sigma2, S=S, beta0=beta0, beta1=beta1)

        except Exception as e:
            return render_template("index.html", error="Error generating data: " + str(e))

    # Populate form with values from session or set default values
    return render_template("index.html",
                           N=session.get('N', 50),
                           mu=session.get('mu', 0),
                           sigma2=session.get('sigma2', 1),
                           S=session.get('S', 100),
                           beta0=session.get('beta0', 0),
                           beta1=session.get('beta1', 1))

# Hypothesis Test Plot
def hypothesis_test_visualization(values, observed_stat, hypothesized_stat, test_param):
    fig, ax = plt.subplots()

    # Plot the histogram of the simulated values (slopes or intercepts)
    ax.hist(values, bins=30, color='lightblue', alpha=0.7)
    
    # Add the observed and hypothesized lines
    ax.axvline(observed_stat, color='red', linestyle='--', label="Observed Value")
    ax.axvline(hypothesized_stat, color='blue', linestyle='-', label="Hypothesized Value")
    
    # Set the plot title based on the test parameter
    ax.set_title(f"Hypothesis Test for {test_param.capitalize()}")  # Dynamic title based on parameter
    ax.set_xlabel(f"{test_param.capitalize()} Value")
    ax.set_ylabel("Frequency")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64


# Confidence Interval Visualization
def confidence_interval_visualization(simulated_values, ci, true_param, mean_estimate, confidence_level, ci_param):
    fig, ax = plt.subplots()
    ax.scatter(simulated_values, [1] * len(simulated_values), color='gray', alpha=0.6, label="Simulated Estimates")
    ax.hlines(1, ci[0], ci[1], color='blue', linewidth=6, label="Confidence Interval")
    ax.scatter(mean_estimate, 1, color='blue', s=100, label="Mean Estimate")
    ax.axvline(true_param, color='red', linestyle='--', linewidth=2, label="True Parameter")
    ax.set_yticks([])
    ax.set_title(f"{int(confidence_level*100)}% Confidence Interval for {ci_param.capitalize()}")
    ax.set_xlabel(f"{ci_param.capitalize()} Value")
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

        # Retrieve slopes or intercepts from session
        slopes = np.array(session.get("slopes", []))
        intercepts = np.array(session.get("intercepts", []))  # Retrieve intercepts from session

        # Debug: Check if slopes and intercepts are being retrieved
        print("Slopes from session:", slopes)
        print("Intercepts from session:", intercepts)

        if len(slopes) == 0 or len(intercepts) == 0:  # Check if both are available
            print("No slopes or intercepts found in session. Redirecting...")
            return redirect(url_for('index'))

        # Determine the observed value and hypothesized value based on test_param
        if test_param == "slope":
            observed_stat = np.mean(slopes)  # Use mean of slopes
            hypothesized_stat = session.get("beta1", 0)  # True slope (beta1)
        elif test_param == "intercept":
            observed_stat = np.mean(intercepts)  # Use mean of intercepts
            hypothesized_stat = session.get("beta0", 0)  # True intercept (beta0)

        # Calculate the standard error of the observed statistic
        if test_param == "slope":
            std_error = np.std(slopes) / np.sqrt(len(slopes))  # Standard error for slopes
        elif test_param == "intercept":
            std_error = np.std(intercepts) / np.sqrt(len(intercepts))  # Standard error for intercepts

        # Calculate the t-statistic
        t_statistic = (observed_stat - hypothesized_stat) / std_error

        # Degrees of freedom
        df = len(slopes) - 1 if test_param == "slope" else len(intercepts) - 1

        # Calculate the p-value using the t-distribution (two-tailed test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))  # Two-tailed p-value

        # Debug: Check p-value calculation
        print(f"P-value: {p_value:.4f}")

        # Generate the hypothesis test plot
        test_plot = hypothesis_test_visualization(slopes if test_param == "slope" else intercepts, observed_stat, hypothesized_stat, test_param)

        return render_template("index.html", 
                               hypothesis_test_result=(
                                   test_param.capitalize(),  # Capitalize the parameter name
                                   f"{observed_stat:.4f}", 
                                   hypothesized_stat, 
                                   f"P-value: {p_value:.4f}"
                               ), 
                               test_plot=test_plot)

    except Exception as e:
        # Debug: Error handling
        print("Error during hypothesis testing:", e)
        return render_template("index.html", error="Error during hypothesis testing: " + str(e))

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    try:
        ci_param = request.form["ci_param"]  # 'slope' or 'intercept'
        confidence_level = float(request.form["confidence_level"])
        
        # Get the correct simulated values based on the ci_param selected
        if ci_param == "slope":
            simulated_values = np.array(session.get("slopes", []))
        elif ci_param == "intercept":
            simulated_values = np.array(session.get("intercepts", []))
        else:
            return redirect(url_for('index'))

        if len(simulated_values) == 0:
            return redirect(url_for('index'))

        # Calculate mean estimate and standard error for confidence interval calculation
        mean_estimate = np.mean(simulated_values)
        std_error = np.std(simulated_values) / np.sqrt(len(simulated_values))
        
        # Confidence interval calculation
        ci = stats.t.interval(confidence_level, len(simulated_values) - 1, loc=mean_estimate, scale=std_error)
        
        # Retrieve the true parameter (slope or intercept) from session
        true_param = session.get("beta1" if ci_param == "slope" else "beta0", 0)
        
        # Check if the true parameter is within the confidence interval
        ci_involves_true = ci[0] <= true_param <= ci[1]

        # Generate the confidence interval visualization
        ci_plot = confidence_interval_visualization(simulated_values, ci, true_param, mean_estimate, confidence_level, ci_param)

        # Round the confidence interval bounds and the mean estimate for display
        confidence_interval_result = (
            ci,
            f"{mean_estimate:.4f}",  # Round mean estimate to 4 decimal places
            ci_involves_true  # Whether the true parameter is inside the CI
        )

        # Render the template with the rounded values
        return render_template("index.html", 
                               confidence_interval_result=confidence_interval_result, 
                               ci_plot=ci_plot,
                               ci_param=ci_param)

    except Exception as e:
        return render_template("index.html", error="Error calculating confidence interval: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
