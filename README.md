# AutoComplete
This project includes a sentence autocomplete feature powered by an N-gram model. Follow the steps below to run the model and see it in action.
## How to Run

1. **Prepare your Data:**
    - Make sure you have your text data ready (e.g., `en_US.twitter.txt`).
    - Run the preprocessing steps if required.

2. **Run the N-gram Autocomplete Script:**
    - Execute the `N-gram.py` script to start the interactive sentence autocomplete feature.

    ```bash
    python N-gram.py
    ```

3. **User Interaction:**
    - When prompted, enter a sequence of words to get the next word suggestion.
    - Optionally, specify a starting letter or sequence for the next word.
    - The program will complete your sentence based on the highest probability suggestion.

## Example Interaction

```bash
Enter a sequence of words: I love
Optional: Enter a starting letter or sequence for the suggested word (or press Enter to skip): y

Completed Sentence:
I love you


## Customization

- **Adjust N-gram Size:** You can modify the N-gram size by adjusting the relevant parameters in the code.
- **Change Smoothing:** Tweak the smoothing parameter `k` to experiment with different probability estimates.



