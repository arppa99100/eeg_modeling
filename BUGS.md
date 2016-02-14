Bugs:
On collecting if you start with click and end with space bar it creates two folders
When wifi goes out in the middle of collecting, it will create a folder without inputs
When the board stops reading but collecting is still going on
When we create a background process to create a model but theres a chance that the eeg.model file has race conditions
Add log tracking to all the process that happen in the background. Never know when an error ocurrs
If we start sampling before collect data finishes we will get race conditions
