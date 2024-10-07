from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index():
    # Python list to populate the dropdown
    options_list = ["Option 1", "Option 2", "Option 3", "Option 4"]

    # Pass the list to the HTML template
    return render_template("index.html", options=options_list)


if __name__ == "__main__":
    app.run(debug=True)
