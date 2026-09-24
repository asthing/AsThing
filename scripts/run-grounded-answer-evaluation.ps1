[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$runId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
$projectName = "asthing-eval-$runId"
$env:EVALUATION_RUN_ID = $runId
$outputDirectory = Join-Path $repositoryRoot ".data\grounded-answer-evaluations\$runId"
$composeArguments = @(
    "compose",
    "--project-name", $projectName,
    "--project-directory", $repositoryRoot,
    "-f", (Join-Path $repositoryRoot "docker-compose.yml"),
    "-f", (Join-Path $repositoryRoot "docker-compose.evaluation.yml")
)

New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

function Invoke-EvaluationCompose {
    param([string[]]$Arguments)

    & docker @composeArguments @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "The isolated evaluation Compose command failed."
    }
}

try {
    & docker volume create asthing_evaluation_embedding_model_cache | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The evaluation embedding cache could not be prepared." }
    & docker volume create asthing_evaluation_ollama_model_cache | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "The evaluation Ollama cache could not be prepared." }

    $configuration = (& docker @composeArguments config --format json) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "The evaluation Compose configuration is invalid." }
$publishedPorts = @(
        $configuration.services.psobject.Properties.Value |
            ForEach-Object {
                if ($_.PSObject.Properties.Name -contains "ports") { $_.ports }
            } |
            Where-Object { $_ }
    )
    if ($publishedPorts.Count -ne 0) {
        throw "The evaluation Compose configuration must not publish service ports."
    }

    Write-Output "Evaluation project: $projectName"
    Write-Output "Evaluation reports: $outputDirectory"
    Invoke-EvaluationCompose @(
        "up", "--build", "--detach", "--wait", "--wait-timeout", "180",
        "postgres", "redis", "minio", "qdrant", "ollama", "api", "worker"
    )
    Invoke-EvaluationCompose @("run", "--rm", "--no-deps", "evaluator")
}
finally {
    & docker @composeArguments down --volumes --remove-orphans | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The isolated evaluation project could not be fully removed."
    }
}
