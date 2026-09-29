/**
 * script.js — AI Spam Message Detector (Frontend Logic)
 *
 * This script handles:
 * 1. Reading the message from the textarea.
 * 2. Sending the message to the Flask API via fetch().
 * 3. Displaying the prediction (Spam / Not Spam) and confidence score.
 * 4. Showing loading, error, and result states.
 * 5. A clear button to reset the form.
 */

// ---------------------------------------------------------------
// GRAB REFERENCES TO HTML ELEMENTS
// ---------------------------------------------------------------
// We get these once so we don't have to search for them every time.

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
// Update the character count below the textarea as the user types.

messageInput.addEventListener('input', () => {
    const length = messageInput.value.length;
    charCount.textContent = `${length} / 5000`;
});


// ---------------------------------------------------------------
// CHECK MESSAGE BUTTON — Main action
// ---------------------------------------------------------------

checkBtn.addEventListener('click', async () => {
    // Step 1: Read the message
    const message = messageInput.value.trim();

    // Step 2: Don't submit if the message is empty
    if (!message) {
        showError('Please enter a message to check.');
        messageInput.focus();
        return;
    }

    // Step 3: Show loading state
    setLoading(true);
    hideResults();
    hideError();

    try {
        // Step 4: Send the message to our Flask API
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        // Step 5: Parse the JSON response from the server
        const data = await response.json();

        // Step 6: Check if the server returned an error
        if (!response.ok || data.error) {
            showError(data.error || 'Something went wrong. Please try again.');
            return;
        }

        // Step 7: Display the prediction result
        showResult(data.prediction, data.confidence);

    } catch (err) {
        // Step 8: Handle network or unexpected errors
        console.error('Fetch error:', err);
        showError('Could not connect to the server. Make sure the Flask app is running.');
    } finally {
        // Always remove the loading state when done
        setLoading(false);
    }
});


// ---------------------------------------------------------------
// CLEAR BUTTON — Reset everything
// ---------------------------------------------------------------

clearBtn.addEventListener('click', () => {
    messageInput.value = '';
    charCount.textContent = '0 / 5000';
    hideResults();
    hideError();
    messageInput.focus();
});


// ---------------------------------------------------------------
// ALLOW CTRL+ENTER TO SUBMIT
// ---------------------------------------------------------------

messageInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        checkBtn.click();
    }
});


// ---------------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------------

/**
 * Show the prediction result in the UI.
 * @param {string} prediction - "Spam" or "Not Spam"
 * @param {number} confidence - Confidence percentage (e.g. 97.42)
 */
function showResult(prediction, confidence) {
    // Determine if the message is spam or safe
    const isSpam = prediction.toLowerCase().includes('spam') &&
                   !prediction.toLowerCase().includes('not');

    // Set the correct visual state (colors, icons)
    resultCard.className = 'result-card ' + (isSpam ? 'spam' : 'safe');
    resultIcon.textContent = isSpam ? '🚫' : '✅';
    resultLabel.textContent = prediction;

    // Show the confidence percentage
    confidenceVal.textContent = `${confidence}%`;

    // Show the section
    resultSection.classList.remove('hidden');

    // Animate the confidence bar (slight delay for visual effect)
    confidenceBar.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confidenceBar.style.width = `${confidence}%`;
        });
    });
}

/**
 * Hide the result section.
 */
function hideResults() {
    resultSection.classList.add('hidden');
    confidenceBar.style.width = '0%';
}

/**
 * Show an error message.
 * @param {string} message - The error text to display
 */
function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

/**
 * Hide the error section.
 */
function hideError() {
    errorSection.classList.add('hidden');
}

/**
 * Toggle the loading state on the Check button.
 * @param {boolean} isLoading - true to show spinner, false to show text
 */
function setLoading(isLoading) {
    if (isLoading) {
        checkBtn.classList.add('loading');
        checkBtn.disabled = true;
    } else {
        checkBtn.classList.remove('loading');
        checkBtn.disabled = false;
    }
}
