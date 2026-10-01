/**
 * script.js — AI Spam Message Detector (Frontend Logic)
 *
 * This script handles:
 * 1. Reading the message from the textarea.
 * 2. Sending the message to the Flask API on Render.
 * 3. Displaying the prediction and confidence score.
 * 4. Showing loading and error states.
 * 5. Clearing the form.
 */


// ---------------------------------------------------------------
// BACKEND URL
// ---------------------------------------------------------------

// Default to the same origin as the Flask app so it works in local
// development without editing this file. You can override it if needed.

const API_URL = window.location.origin || "http://127.0.0.1:5000";


// ---------------------------------------------------------------
// GRAB REFERENCES TO HTML ELEMENTS
// ---------------------------------------------------------------

const messageInput   = document.getElementById('message-input');
const charCount      = document.getElementById('char-count');
const checkBtn       = document.getElementById('check-btn');
const clearBtn       = document.getElementById('clear-btn');
const resultSection  = document.getElementById('result-section');
const resultCard     = document.getElementById('result-card');
const resultBadge    = document.getElementById('result-badge');
const resultIcon     = document.getElementById('result-icon');
const resultLabel    = document.getElementById('result-label');
const confidenceBar  = document.getElementById('confidence-bar');
const confidenceVal  = document.getElementById('confidence-value');
const errorSection   = document.getElementById('error-section');
const errorMessage   = document.getElementById('error-message');


// ---------------------------------------------------------------
// CHARACTER COUNTER
// ---------------------------------------------------------------

messageInput.addEventListener('input', () => {

    const length = messageInput.value.length;

    charCount.textContent = `${length} / 5000`;

});


// ---------------------------------------------------------------
// CHECK MESSAGE BUTTON
// ---------------------------------------------------------------

checkBtn.addEventListener('click', async () => {

    // Read the message entered by the user.
    const message = messageInput.value.trim();


    // Don't continue if the message is empty.
    if (!message) {

        showError('Please enter a message to check.');

        messageInput.focus();

        return;
    }


    // Show loading state.
    setLoading(true);

    hideResults();

    hideError();


    try {

        // -------------------------------------------------------
        // SEND MESSAGE TO RENDER BACKEND
        // -------------------------------------------------------

        const response = await fetch(`${API_URL}/predict`, {

            method: 'POST',

            headers: {
                'Content-Type': 'application/json'
            },

            body: JSON.stringify({
                message: message
            })

        });


        // Convert server response into JSON.
        const data = await response.json();


        // Check if backend returned an error.
        if (!response.ok || data.error) {

            showError(
                data.error ||
                'Something went wrong. Please try again.'
            );

            return;
        }


        // Display the prediction.
        showResult(
            data.prediction,
            data.confidence
        );


    } catch (err) {

        // Show error in browser console.
        console.error('Fetch error:', err);


        // Display user-friendly error.
        showError(
            'Could not connect to the server. Please check the Render backend.'
        );


    } finally {

        // Remove loading state.
        setLoading(false);

    }

});


// ---------------------------------------------------------------
// CLEAR BUTTON
// ---------------------------------------------------------------

clearBtn.addEventListener('click', () => {

    messageInput.value = '';

    charCount.textContent = '0 / 5000';

    hideResults();

    hideError();

    messageInput.focus();

});


// ---------------------------------------------------------------
// CTRL + ENTER TO CHECK MESSAGE
// ---------------------------------------------------------------

messageInput.addEventListener('keydown', (e) => {

    if (
        (e.ctrlKey || e.metaKey) &&
        e.key === 'Enter'
    ) {

        checkBtn.click();

    }

});


// ---------------------------------------------------------------
// SHOW RESULT
// ---------------------------------------------------------------

function showResult(prediction, confidence) {

    // Convert prediction to lowercase for checking.
    const predictionText = prediction.toLowerCase();


    // Check whether prediction is Spam.
    const isSpam =
        predictionText.includes('spam') &&
        !predictionText.includes('not');


    // Set the result card style.
    resultCard.className =
        'result-card ' +
        (isSpam ? 'spam' : 'safe');


    // Set result icon.
    resultIcon.textContent =
        isSpam ? '🚫' : '✅';


    // Set result text.
    resultLabel.textContent = prediction;


    // Display confidence.
    confidenceVal.textContent =
        `${confidence}%`;


    // Show result section.
    resultSection.classList.remove('hidden');


    // Start confidence bar from 0%.
    confidenceBar.style.width = '0%';


    // Animate confidence bar.
    requestAnimationFrame(() => {

        requestAnimationFrame(() => {

            confidenceBar.style.width =
                `${confidence}%`;

        });

    });

}


// ---------------------------------------------------------------
// HIDE RESULTS
// ---------------------------------------------------------------

function hideResults() {

    resultSection.classList.add('hidden');

    confidenceBar.style.width = '0%';

}


// ---------------------------------------------------------------
// SHOW ERROR
// ---------------------------------------------------------------

function showError(message) {

    errorMessage.textContent = message;

    errorSection.classList.remove('hidden');

}


// ---------------------------------------------------------------
// HIDE ERROR
// ---------------------------------------------------------------

function hideError() {

    errorSection.classList.add('hidden');

}


// ---------------------------------------------------------------
// LOADING STATE
// ---------------------------------------------------------------

function setLoading(isLoading) {

    if (isLoading) {

        checkBtn.classList.add('loading');

        checkBtn.disabled = true;

    } else {

        checkBtn.classList.remove('loading');

        checkBtn.disabled = false;

    }

}