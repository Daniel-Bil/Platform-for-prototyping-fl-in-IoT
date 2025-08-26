# Set paths
$SourcePath = "D:\user\PycharmProjects\Platform-for-prototyping-fl-in-IoT"
$DestinationZip = "D:\user\PycharmProjects\Platform-for-prototyping-fl-in-IoT_clean.zip"

# Define patterns to exclude (add/remove as needed)
$ExcludePatterns = @(
    "venv",                # Virtual environment
    "__pycache__",         # Python cache
    "*.pyc",               # Compiled Python files
    "*.pyo",
    "*.log",               # Logs
    ".idea",               # PyCharm project files
    "*.tmp",               # Temp files
    "requirements39.txt"   # Old requirements file
)

# Create a temporary folder to copy filtered content
$TempFolder = "$env:TEMP\ProjectTempCopy"
if (Test-Path $TempFolder) { Remove-Item $TempFolder -Recurse -Force }
New-Item -ItemType Directory -Path $TempFolder | Out-Null

# Copy files excluding unwanted patterns
Get-ChildItem -Path $SourcePath -Recurse | Where-Object {
    $RelativePath = $_.FullName.Substring($SourcePath.Length + 1)
    foreach ($Pattern in $ExcludePatterns) {
        if ($RelativePath -like "*$Pattern*") { return $false }
    }
    return $true
} | ForEach-Object {
    $Target = Join-Path $TempFolder $_.FullName.Substring($SourcePath.Length + 1)
    if ($_.PSIsContainer) {
        New-Item -ItemType Directory -Path $Target -Force | Out-Null
    } else {
        Copy-Item $_.FullName -Destination $Target -Force
    }
}

# Create ZIP from filtered content
if (Test-Path $DestinationZip) { Remove-Item $DestinationZip -Force }
Compress-Archive -Path "$TempFolder\*" -DestinationPath $DestinationZip

# Cleanup
Remove-Item $TempFolder -Recurse -Force

Write-Host "Project successfully zipped to: $DestinationZip"
