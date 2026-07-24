param($f)
$content = Get-Content $f
$out = @()
foreach ($line in $content) {
    if ($line -match '^pick c1df266') {
        $out += $line -replace '^pick ', 'fixup '
    } else {
        $out += $line
    }
}
Set-Content $f $out
