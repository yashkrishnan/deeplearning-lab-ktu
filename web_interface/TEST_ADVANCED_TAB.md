# Advanced Tab Testing Guide

## Quick Test Steps

### 1. Start the Server
```bash
cd web_interface
python3 app.py
```

### 2. Open Browser
Navigate to: http://localhost:5001

### 3. Click Advanced Tab
The fourth tab in the navigation

### 4. Test Confirmation Dialog
Click any task card (e.g., "Install Minimal Requirements")

**Expected Result**:
- Confirmation dialog appears
- Shows task name, duration, and warning
- Has OK and Cancel buttons

### 5. Test Task Execution
Click OK in the confirmation dialog

**Expected Result**:
- Console shows "Starting task..." message
- Status changes from "Ready" to "Running"
- Console shows command being executed
- Console shows note about checking terminal

### 6. Test Status Persistence
- Navigate to another tab (e.g., "Full Labs")
- Navigate back to "Advanced" tab

**Expected Result**:
- Task status is still "Running" (if task is still executing)
- Status persists across navigation

### 7. Test Interactive Task Warning
Click "Interactive Dataset Download" task

**Expected Result**:
- Confirmation dialog shows "Varies" for duration
- Warning mentions terminal input requirement
- After confirming, console shows: "⚠️ INTERACTIVE TASK: Please check your terminal and provide input as needed."

## Troubleshooting

### Buttons Not Responding
1. Open browser console (F12)
2. Check for JavaScript errors
3. Verify Flask server is running
4. Refresh the page (Ctrl+R or Cmd+R)

### Status Not Updating
1. Check browser console for fetch errors
2. Verify `/api/setup/statuses` endpoint is accessible
3. Check Flask server logs for errors

### Confirmation Dialog Not Appearing
1. Check browser console for JavaScript errors
2. Verify `confirmSetupTask()` function is defined
3. Check if browser is blocking dialogs

## Manual API Testing

Test the backend directly:

```bash
# Get all task statuses
curl http://localhost:5001/api/setup/statuses

# Start a task (this will actually run the command!)
curl -X POST http://localhost:5001/api/setup/install-minimal
```

## Expected Console Output

When clicking a task, the console should show:
```
Starting task: install-minimal...
Installing minimal requirements started successfully
Command: python3 -m pip install -r requirements-minimal.txt
Note: This task runs in the background. Check terminal for progress.
```

For interactive tasks:
```
Starting task: download-interactive...
Interactive dataset download started successfully
Command: bash download_datasets_interactive.sh
⚠️ INTERACTIVE TASK: Please check your terminal and provide input as needed.
Note: This task runs in the background. Check terminal for progress.
```

## Status Indicators

- **Ready**: Gray/purple, task not started
- **Running**: Pink/red with pulsing animation
- **Completed**: Blue, task finished successfully
- **Failed**: Red, task finished with error
- **Error**: Red, exception occurred

## Common Issues

### Issue: "Flask module not found"
**Solution**: Install Flask
```bash
pip install flask
```

### Issue: "Permission denied" for shell scripts
**Solution**: Make scripts executable
```bash
chmod +x download_all_datasets.sh
chmod +x run_all_labs.sh
chmod +x labs_lite/run_lite_labs.sh
```

### Issue: Task status stuck on "Running"
**Cause**: Task may have completed but status not updated
**Solution**: 
1. Check Flask terminal for task completion
2. Refresh the page
3. Status should update on next poll

## Success Criteria

✅ Confirmation dialog appears before task execution
✅ Duration and warnings displayed correctly
✅ Task starts after confirmation
✅ Status changes to "Running"
✅ Console shows task details
✅ Status persists across page navigation
✅ Interactive tasks show special warning
✅ Status updates automatically every 3 seconds
✅ Cannot start same task twice
✅ Status colors match task state