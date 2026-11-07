[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$runId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
$projectName = "asthing-diagnostic-$runId"
$previousRunId = $env:EVALUATION_RUN_ID
$env:EVALUATION_RUN_ID = $runId
$outputDirectory = Join-Path $repositoryRoot ".data\grounded-answer-diagnostics\$runId"
$composeArguments = @(
    "compose", "--project-name", $projectName, "--project-directory", $repositoryRoot,
    "-f", (Join-Path $repositoryRoot "docker-compose.yml"),
    "-f", (Join-Path $repositoryRoot "docker-compose.evaluation.yml"),
    "-f", (Join-Path $repositoryRoot "docker-compose.diagnostics.yml")
)

function Invoke-DiagnosticCompose {
    param([string[]]$Arguments)
    # Native build/startup errors can include infrastructure details. Keep them off the console.
    & docker @composeArguments @Arguments *> $null
    if ($LASTEXITCODE -ne 0) { throw "The isolated diagnostic command failed safely." }
}

try {
    # Existing evaluation caches only. Never download or fall back to default-stack caches.
    foreach ($cache in @("asthing_evaluation_embedding_model_cache", "asthing_evaluation_ollama_model_cache")) {
        & docker volume inspect $cache *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Diagnostic preflight failed. Run the documented evaluation prewarm commands."
        }
    }
    $configuration = (& docker @composeArguments config --format json 2>$null) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "The diagnostic Compose configuration is invalid." }
    foreach ($service in $configuration.services.psobject.Properties.Value) {
        if (($service.PSObject.Properties.Name -contains "ports") -and @($service.ports).Count -ne 0) {
            throw "The diagnostic configuration must not publish ports."
        }
    }
    if ($configuration.services.evaluator.logging.driver -ne "none") {
        throw "Diagnostic container logging must be disabled."
    }
    $fixtures = @($configuration.services.evaluator.volumes | Where-Object { $_.target -eq "/evaluation-fixtures" })
    if ($fixtures.Count -ne 1 -or -not $fixtures[0].read_only) {
        throw "Diagnostic fixtures must be read-only."
    }
    Write-Output "Diagnostic project: $projectName"
    Write-Output "Diagnostic artifacts: $outputDirectory"
    Write-Output "Configuration verified: zero published ports; read-only fixtures; diagnostic logging disabled."
    Invoke-DiagnosticCompose @(
        "up", "--build", "--detach", "--wait", "--wait-timeout", "180",
        "postgres", "redis", "minio", "qdrant", "ollama", "api", "worker"
    )
    Write-Output "Isolated diagnostic services are ready."
    Invoke-DiagnosticCompose @("run", "--build", "--rm", "--no-deps", "evaluator")
    Write-Output "Diagnostic execution completed. Inspect the safe decision summary."
}
finally {
    & docker @composeArguments down --volumes --remove-orphans *> $null
    $cleanupExitCode = $LASTEXITCODE
    $env:EVALUATION_RUN_ID = $previousRunId
    if ($cleanupExitCode -ne 0) { throw "The isolated diagnostic project cleanup failed." }
    Write-Output "Disposable diagnostic data removed; dedicated evaluation caches preserved."
}
