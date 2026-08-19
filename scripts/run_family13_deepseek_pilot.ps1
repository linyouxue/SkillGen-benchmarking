param(
    [switch]$Execute,
    [switch]$Resume,
    [switch]$AuthorizeBudgetPolicyAmendment,
    [string]$ExpectedNewProtocolHash
)

$ErrorActionPreference = 'Stop'
if ($Resume -and -not $Execute) {
    throw '-Resume requires -Execute.'
}
if ($AuthorizeBudgetPolicyAmendment) {
    if (-not $Execute -or -not $Resume -or -not $ExpectedNewProtocolHash) {
        throw '-AuthorizeBudgetPolicyAmendment requires -Execute, -Resume, and -ExpectedNewProtocolHash.'
    }
}
elseif ($ExpectedNewProtocolHash) {
    throw '-ExpectedNewProtocolHash requires -AuthorizeBudgetPolicyAmendment.'
}

$bashLauncher = Join-Path $PSScriptRoot 'run_family13_deepseek_pilot.sh'
if (-not (Test-Path -LiteralPath $bashLauncher)) {
    throw "Missing launcher: $bashLauncher"
}

$oldDeepSeek = $env:DEEPSEEK_API_KEY
$oldOpenAI = $env:OPENAI_API_KEY
$oldWslEnv = $env:WSLENV

try {
    if (-not $env:OPENAI_API_KEY) {
        $env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable(
            'OPENAI_API_KEY',
            'User'
        )
    }
    if ($Execute -and -not $env:OPENAI_API_KEY) {
        throw 'OPENAI_API_KEY is required for the original SkillGen embedding step.'
    }

    if ($Execute -and -not $env:DEEPSEEK_API_KEY) {
        $secure = Read-Host 'DeepSeek official API key' -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try {
            $env:DEEPSEEK_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR(
                $pointer
            )
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }

    $forward = @()
    if ($env:WSLENV) {
        $forward += $env:WSLENV -split ':' | Where-Object { $_ }
    }
    foreach ($name in @(
        'DEEPSEEK_API_KEY',
        'OPENAI_API_KEY',
        'SKILLGEN_PILOT_RUN_ROOT'
    )) {
        if ($forward -notcontains $name) {
            $forward += $name
        }
    }
    $env:WSLENV = $forward -join ':'

    $resolvedLauncher = (Resolve-Path -LiteralPath $bashLauncher).Path
    if ($resolvedLauncher -notmatch '^[A-Za-z]:\\') {
        throw "Expected an absolute Windows launcher path: $resolvedLauncher"
    }
    $drive = $resolvedLauncher.Substring(0, 1).ToLowerInvariant()
    $relative = $resolvedLauncher.Substring(3).Replace('\', '/')
    $wslScript = "/mnt/$drive/$relative"

    $arguments = @('-d', 'Ubuntu', '--', 'bash', $wslScript)
    if ($Execute) {
        $arguments += '--execute'
    }
    if ($Resume) {
        $arguments += '--resume'
    }
    if ($AuthorizeBudgetPolicyAmendment) {
        $arguments += '--authorize-budget-policy-amendment'
        $arguments += '--expected-new-protocol-hash'
        $arguments += $ExpectedNewProtocolHash
    }
    & wsl.exe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Pilot launcher exited with code $LASTEXITCODE."
    }
}
finally {
    $env:DEEPSEEK_API_KEY = $oldDeepSeek
    $env:OPENAI_API_KEY = $oldOpenAI
    $env:WSLENV = $oldWslEnv
}
