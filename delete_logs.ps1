# Ce script supprime tous les fichiers et sous-dossiers dans 'logs/fit/'

# Chemin relatif au dossier 'logs/fit/' à partir de la racine du projet
$logPath = "./logs/fit/"

# Vérifie les dossiers et supprime leur contenu
if (Test-Path $logPath) {
    Get-ChildItem -Path $logPath -Recurse | Remove-Item -Force -Recurse
    Write-Output "Tous les logs ont été supprimés de $logPath."
} else {
    Write-Output "Le chemin spécifié '$logPath' n'existe pas."
}
