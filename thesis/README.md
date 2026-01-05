# Ein Beispiel LaTeX Dokument

Dies ist ein einfaches LaTeX-Dokument, das grundlegende Funktionen von LaTeX demonstriert, wie etwa das Setzen von Textformatierungen, das Erstellen von mathematischen Formeln, das Erstellen von Listen, Tabellen und das Einfügen von Bildern.
## Repository klonen

Um dieses Repository zu klonen, führe folgenden Befehl in deinem Terminal aus:

```bash
git clone https://github.com/mcfiet/latex-example.git
```

## Anforderungen

- **LaTeX-Distribution**: Dieses Projekt verwendet `TeX Live`. Um das Projekt zu kompilieren, stelle sicher, dass du die folgenden Pakete installiert hast:
  - `texlive-core`
  - `texlive-science`
  - `texlive-bin`
  - `texlive-langextra`
  - `texlive-fontsextra`
  - `texlab`
  - `zathura`
  - `zathura-pdf-poppler`
  - `texlive-latexmk`
  - `texlive-lang`
  - `texlive-bibtexextra`
  - `texlive-xcolor`

Falls du `pacman` verwendest, kannst du diese Pakete mit folgendem Befehl installieren:

```bash
sudo pacman -S texlive-core texlive-science texlive-bin texlive-langextra texlive-fontsextra texlab zathura zathura-pdf-poppler texlive-latexmk texlive-lang texlive-bibtexextra texlive-xcolor texlive-biblatex 
```

## Verwendung

1. Stelle sicher, dass alle benötigten Pakete installiert sind.
2. Öffne eine Terminal-Sitzung und navigiere zu dem Ordner, der das LaTeX-Dokument enthält.
3. Führe den folgenden Befehl aus, um das LaTeX-Dokument zu kompilieren und ein PDF zu erstellen:

```bash
pdflatex beispiel_dokument.tex
```

4. Öffne das generierte PDF mit deinem bevorzugten PDF-Viewer (z. B. `zathura`):

```bash
zathura beispiel_dokument.pdf
```

## Struktur des Projekts

- `beispiel_dokument.tex`: Das Haupt-LaTeX-Dokument.
- `bild.jpg`: Ein Beispielbild, das in das Dokument eingefügt wird.

## Inhalt des Dokuments

Das Dokument zeigt, wie man folgende Dinge in LaTeX umsetzt:

- **Textformatierungen**: Fett (`\textbf{}`), Kursiv (`\textit{}`).
- **Mathematische Formeln**: Inline-Formeln und Gleichungen in Blöcken.
- **Listen**: Ungeordnete und geordnete Listen.
- **Tabellen**: Einfache und farbige Tabellen mit Kopfzeilen.
- **Bilder und Figuren**: Ein Bild wird eingefügt mit einer `caption`.
