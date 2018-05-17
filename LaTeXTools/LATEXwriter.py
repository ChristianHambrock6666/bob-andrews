import os, sys
import subprocess
from subprocess import call
import pandas
import numpy as np

'''
Translated from Matlab
'''


class LATEXwriter:
    locURL = ""
    outputFile = ""
    working_directory = ""
    destinationFileName = ""
    templateFileName = ""
    fontSize = 10
    numberFigures = 0
    typeface = "lmodern"  # typefaces of LaTeX usable (helvet, courier, times,...) whatever packages are on the market

    keyValueReplacementList = None
    usedKeys = None  # to know the order in which the keys are included, since it seems the coidctionary is not alwys consistently ordered.

    templateString = ""

    style = "Standard"

    ColorMap = None

    def __init__(o,
                 in_working_directory,
                 in_destination_file_name,
                 templatefile='',
                 style='Standard',
                 typeface='lmodern',
                 doc_name='doc name',
                 font_size=10
                 ):

        o.working_directory = in_working_directory
        o.destinationFileName = in_destination_file_name
        o.locURL = o.working_directory + '/' + o.destinationFileName + '.tex'
        o.outputFile = o.working_directory + '/' + o.destinationFileName + '.pdf'
        o.templateFileName = templatefile
        o.style = style
        o.typeface = typeface
        o.docName = doc_name
        o.fontSize = font_size

        o.keyValueReplacementList = dict()  # containers.Map(keySet,valueSet)
        o.usedKeys = []
        o.ColorMap = dict()

        o.addColor("$Yellow", [0.95, 0.8, 0.0])
        o.addColor("$Green", [0.1, 0.95, 0.0])
        o.addColor("$Red", [0.95, 0.1, 0.0])
        o.addColor("$Orange", [0.9, 0.6, 0.0])
        o.addColor("$Magenta", [0.9, 0.1, 0.7])
        o.addColor("$Gray", [0.6, 0.6, 0.6])
        o.addColor("$Black", [0.0, 0.0, 0.0])
        o.addColor("$Cyan", [0.1, 0.7, 0.9])

    def addText(o, inText, key='StandardParseBODYKEY'):
        """
        Because its LaTeX everything boils down to adding text at
        the correct position. So each method will just prepare the text and
        use ultimately THIS method to queue it in the insertion queue via the
        dictionary KEY VALUE combination.

        :param inText:
        :param key:
        :return:
        """

        if key in o.keyValueReplacementList:
            # if o.KeyValueReplacementList.has_key(key):
            o.keyValueReplacementList[key] = o.keyValueReplacementList[key] + inText
        else:
            o.keyValueReplacementList[key] = inText
            o.usedKeys.append(key)

    def includeTitlepage(o, key='StandardParseBODYKEY', title='', explanation='', figurePath=''):
        """
        Includes a titlepage.

        :param key:
        :param title:
        :param explanation:
        :param figurePath:
        :return:
        """

        tmpString = ' \n' \
                    '\clearpage\n' \
                    '\\newcommand\\nbvspace[1][3]{\\vspace*{\stretch{#1}}}\n' \
                    '\\newcommand\\nbstretchyspace{\spaceskip0.5em plus 0.25em minus 0.25em}\n' \
                    '\\newcommand{\\nbtitlestretch}{\spaceskip0.6em}\n' \
                    '\\begin{center}\n' \
                    '\\bfseries\n' \
                    '\\nbvspace[1]\n' \
                    '\\Huge\n' \
                    '{\\nbtitlestretch\huge\n' \
                    'TITLEKEY }\n' \
                    '\\nbvspace[1]\n' \
                    '\\normalsize \\begin{center}\n' \
                    'EXPLANATIONKEY\n' \
                    '\end{center}\\nbvspace[2]\n' \
                    ' \n' \
                    'TITLEFIGUREKEY\n' \
                    '\\nbvspace[3]\n' \
                    '\\normalsize\n' \
                    ' \n' \
                    '\large\n' \
                    '\\nbvspace[1]\n' \
                    '\end{center} '

        tmpString = tmpString.replace('TITLEKEY', title)
        tmpString = tmpString.replace('EXPLANATIONKEY', explanation)
        if figurePath == '':
            tmpString = tmpString.replace('TITLEFIGUREKEY', figurePath)
        else:
            tmpString = tmpString.replace('TITLEFIGUREKEY',
                                          '\includegraphics[width=0.8\\textwidth, angle=0]{TITLEFIGUREKEY}')
            tmpString = tmpString.replace('TITLEFIGUREKEY', figurePath)

        o.addText(tmpString, key=key)

    def includeGraphics(o, inPath, key='StandardParseBODYKEY', caption='', figureType='figure', textwidth=0.8):
        """
        figureType = 'figure', 'sidewaysfigure'

        :param inPath:
        :param key:
        :param caption:
        :param figureType:
        :param textwidth:
        :return:
        """

        VALUESfigure = inPath

        locFigureString = '\\begin{figureTypeKEY}[H]\n' \
                          '\\begin{center}\n' \
                          '\\includegraphics[width=' + str(textwidth) + '\\textwidth, angle=0]{KEYfigure}\n' \
                                                                        '\\end{center}\n' \
                                                                        '\\caption{KEYcaption}\n' \
                                                                        '\\end{figureTypeKEY}\n'

        locFigureString = locFigureString.replace('KEYfigure', VALUESfigure)
        locFigureString = locFigureString.replace('figureTypeKEY', figureType)
        locFigureString = locFigureString.replace('KEYcaption', caption)

        o.addText(locFigureString, key=key)

    def addFigure(o, fig, key='StandardParseBODYKEY', caption='', textwidth=0.8, where='H', figureType='figure',
                  sidewaysfigure=False):
        """
        takes the last figure, stores it as jpg in a temporary folder and
        later includes it in the latex document via the given name (made from the keyword).
        figureType = 'figure' || 'sidewaysfigure'
        TODO: allow a list of figures to be entered and add them in one environment.

        :param fig:
        :param key:
        :param caption:
        :param textwidth:
        :param where:
        :param figureType:
        :param sidewaysfigure:
        :return:
        """
        if sidewaysfigure:
            figureType = "sidewaysfigure"
        locFigurePath = o.working_directory + '/TMPfigures/'

        if not os.path.exists(locFigurePath):
            os.makedirs(locFigurePath)

        o.numberFigures = o.numberFigures + 1

        VALUESfigure = './TMPfigures/' + key + str(o.numberFigures) + '.png'

        fig.savefig(o.working_directory + '/TMPfigures/' + key + str(o.numberFigures) + '.png')

        locFigureString = '\\begin{figure}[' + where + ']\n' \
                                                       '\\begin{center}\n' \
                                                       '\includegraphics[width=' + str(
            textwidth) + '\\textwidth, angle=0]{KEYfigure}\n' \
                         '\end{center}\n' \
                         '\caption{KEYcaption}\n' \
                         '\end{figure}\n'

        if figureType.lower() == "sidewaysfigure":
            locFigureString = '\\begin{landscape}\n' + locFigureString + '\end{landscape}\n'

        locFigureString = locFigureString.replace('KEYfigure', VALUESfigure)
        locFigureString = locFigureString.replace('figureTypeKEY', figureType)
        locFigureString = locFigureString.replace('KEYcaption', caption)

        o.addText(locFigureString, key=key)

    def addSection(o, inText, key='StandardParseBODYKEY'):
        """
        Adds a section.
        """
        locSectionString = '\section{' + inText + '}\n'
        o.addText(locSectionString, key=key)

    def addTable(o, inMatrix, key='StandardParseBODYKEY', fontsize=10, lineSeparation=8, caption='', sideways=False,
                 verb_sep="♥", alignment="l", literalColumns=[]):
        """
        Adds latex table
        """

        if isinstance(inMatrix, pandas.core.frame.DataFrame):
            columnNames = inMatrix._info_axis.values
            myValues = [[str(entry) + " " for entry in row] for row in inMatrix.values]
            inStringMatrix = np.vstack(([np.array(columnNames)], np.array(myValues)))

        else:
            inStringMatrix = np.row_stack(inMatrix)

        nr_rows_p, nr_cols_p = inStringMatrix.shape

        if literalColumns == []:
            literalColumns = [True for i in range(nr_cols_p)]

        if nr_rows_p * nr_cols_p == 0:
            print('Error: empty')
            return

        VALUEbody = ''
        VALUEcolumnorganization = ''
        VALUEheadline = ''

        locTableString = '{\\fontsize{' + str(fontsize) + '}{' + str(lineSeparation) + 'mm} \\selectfont  STARTKEY\n' \
                                                                                       '\\begin{longtable}{KEYcolumnorganization} \caption{CaptionKEY}\n ' \
                                                                                       '\\\\ \hline \n KEYHEADLINE \\\\ \\hline\\hline  \n' \
                                                                                       '\\endfirsthead\n' \
                                                                                       '   \\hline \n KEYHEADLINE \\\\ \\hline\\hline  \n' \
                                                                                       '\\endhead\n' \
                                                                                       '\\hline  \n' \
                                                                                       'KEYTABLEBODY' \
                                                                                       '\\hline\n' \
                                                                                       '\\end{longtable}  STOPKEY }\n\n'

        VALUEcolumnorganization = VALUEcolumnorganization + '|' + alignment
        for c in range(0, nr_cols_p - 1):
            VALUEcolumnorganization = VALUEcolumnorganization + '|' + alignment
        VALUEcolumnorganization = VALUEcolumnorganization + '|'

        for c in range(0, nr_cols_p - 1):
            VALUEheadline = VALUEheadline + inStringMatrix[0, c] + ' & '

        VALUEheadline = VALUEheadline + inStringMatrix[0, nr_cols_p - 1] + ''

        for r in range(1, nr_rows_p):
            for c in range(0, nr_cols_p):
                if literalColumns[c]:
                    VALUEbody = VALUEbody + "\\verb" + verb_sep + inStringMatrix[r, c] + verb_sep
                else:
                    VALUEbody = VALUEbody + inStringMatrix[r, c]
                if not c == nr_cols_p - 1:
                    VALUEbody = VALUEbody + ' & '
                else:
                    VALUEbody = VALUEbody + ' \\\\ \hline  \n'

        locTableString = locTableString.replace('KEYcolumnorganization', VALUEcolumnorganization)
        locTableString = locTableString.replace('KEYHEADLINE', VALUEheadline)
        locTableString = locTableString.replace('KEYTABLEBODY', VALUEbody)
        locTableString = locTableString.replace('CaptionKEY', caption)
        if sideways:  # insert landscape mode but leave START STOP for possible further insertions
            locTableString = locTableString.replace('STARTKEY', '\\begin{landscape} STARTKEY')
            locTableString = locTableString.replace('STOPKEY', 'STOPKEY \\end{landscape}')

        locTableString = locTableString.replace('STARTKEY', '')
        locTableString = locTableString.replace('STOPKEY', '')

        o.addText(locTableString, key=key)

    def includeTOC(o, key='StandardParseBODYKEY'):
        """
        Includes table of contents (recommended to be at the beginning, as it is parsed where it is executed, if no key is given)

        :param key:
        :return:
        """
        o.addText('\\tableofcontents \n', key=key)

    def standardLayout(o):
        """

        :return:
        """

        retText = '\documentclass[a4paper,' + str(o.fontSize) + 'pt,twoside]{article} \n' \
                                                                '\\usepackage[headsep=2.5cm,headheight=2cm]{geometry}\n\\geometry{a4paper, left=25mm, top=50mm, bottom=20mm, right=25mm}\n' \
                                                                '\\usepackage[absolute]{textpos}\n' \
                                                                '\\usepackage{amsmath}\n' \
                                                                '\\usepackage{amssymb}\n' \
                                                                '\\usepackage{bm}\n' \
                                                                '\\usepackage{calc}\n' \
                                                                '\\usepackage{caption}\n' \
                                                                '\\usepackage{' + o.typeface + '}\n' \
                                                                                               '\\usepackage[english]{babel}\n' \
                                                                                               '\\usepackage{epstopdf}\n' \
                                                                                               '\\usepackage[export]{adjustbox}\n' \
                                                                                               '\\usepackage{fancyhdr}\n\\pagestyle{fancy}\n\\fancyhf{} ' \
                                                                                               '\\renewcommand{\headrulewidth}{1pt} ' \
                                                                                               '\\fancyfoot[C]{\\thepage}' \
                                                                                               '\\usepackage{fixltx2e}\n' \
                                                                                               '\\usepackage{float}\n' \
                                                                                               '\\usepackage{framed}\n' \
                                                                                               '\\usepackage{geometry}\n' \
                                                                                               '\\usepackage{graphicx}\n' \
                                                                                               '\\usepackage{hyperref}\n' \
                                                                                               '\\usepackage{listings}\n' \
                                                                                               '\\usepackage{longtable}\n' \
                                                                                               '\\usepackage{lscape}\n' \
                                                                                               '\\usepackage{makeidx}\n' \
                                                                                               '\\usepackage{multicol}\n' \
                                                                                               '\\usepackage{multirow}\n' \
                                                                                               '\\usepackage[parfill]{parskip}\n' \
                                                                                               '\\usepackage{rotating}\n' \
                                                                                               '\\usepackage{sectsty}\n' \
                                                                                               '\\usepackage[table]{xcolor}\n' \
                                                                                               '\\usepackage{textcomp}\n' \
                                                                                               '\\usepackage[utf8]{inputenc}\n' \
                                                                                               '\\usepackage{xcolor}\n' \
                                                                                               '\\usepackage{pgfplots}\n' \
                                                                                               '\\usepackage[explicit]{titlesec}\n' \
                                                                                               '\\renewcommand{\\familydefault}{\sfdefault}\n' \
                                                                                               '\\usepgfplotslibrary{external} \n' \
                                                                                               '\\tikzexternalize\n' \
                                                                                               '\\usepgfplotslibrary{dateplot}\n' \
                                                                                               '\\usepgfplotslibrary{fillbetween}\n' \
                                                                                               '\captionsetup{labelsep=colon,justification=centering,labelfont=bf,singlelinecheck=off,skip=4pt,position=top}\n' \
                                                                                               '\def\\transformtime#1:#2!{\n' \
                                                                                               '\pgfkeys{/pgf/fpu=true,/pgf/fpu/output format=fixed}\n' \
                                                                                               '\pgfmathparse{#1*3600-\pgfkeysvalueof{/pgfplots/timeplot zero}*3600+#2*60}\n' \
                                                                                               '\pgfkeys{/pgf/fpu=false}\n' \
                                                                                               '}\n' \
                                                                                               '\pgfplotsset{\n' \
                                                                                               'timeplot zero/.initial=0,\n' \
                                                                                               'timeplot/.style={\n' \
                                                                                               'y coord trafo/.code={\expandafter\\transformtime##1!},\n' \
                                                                                               'y coord inv trafo/.code={%\n' \
                                                                                               '\pgfkeys{/pgf/fpu=true,/pgf/fpu/output format=fixed,}\n' \
                                                                                               '\pgfmathsetmacro\hours{floor(##1/3600)+\pgfkeysvalueof{/pgfplots/timeplot zero}}\n' \
                                                                                               '\pgfmathsetmacro\minutes{floor((##1-(\hours - \pgfkeysvalueof{/pgfplots/timeplot zero})*3600)/60)}\n' \
                                                                                               '\def\pgfmathresult{\n' \
                                                                                               '\pgfmathprintnumber{\hours}:%\n' \
                                                                                               '\pgfmathparse{int(mod(\minutes,60))/100}%\n' \
                                                                                               '\pgfmathprintnumber[skip 0.=true, dec sep={}, fixed]{\pgfmathresult}\n' \
                                                                                               '}\n' \
                                                                                               '\pgfkeys{/pgf/fpu=false}\n' \
                                                                                               '},\n' \
                                                                                               'scaled y ticks=false,\n' \
                                                                                               'yticklabel=\tick\n' \
                                                                                               '}\n' \
                                                                                               '}\n' \
                                                                                               ' \n' \
                                                                                               '\pgfplotsset{%\n' \
                                                                                               ',compat=1.13\n' \
                                                                                               ',colormap={Green}{rgb255(0cm)=(0,0,0); rgb255(1cm)=(0,255,0)}\n' \
                                                                                               ',colormap={Blue}{rgb255(0cm)=(0,0,0); rgb255(1cm)=(0,0,255)}\n' \
                                                                                               ',colormap={Red}{rgb255(0cm)=(0,0,0); rgb255(1cm)=(255,0,0)}\n' \
                                                                                               '} \n' \
                                                                                               ' \n' \
                                                                                               'KEYDefinedColors\n' \
                                                                                               ' \n' \
                                                                                               '\\newcommand*{\mcol}{}\n' \
                                                                                               '\def\mcol#1#{\mcolaux{#1}}\n' \
                                                                                               '\\newcommand*{\mcolaux}[3]{\n' \
                                                                                               '  \protect\leavevmode\n' \
                                                                                               '  \\begingroup\n' \
                                                                                               '    \color#1{#2}#3\n' \
                                                                                               '  \endgroup\n' \
                                                                                               '}\n' \
                                                                                               ' \n' \
                                                                                               '\\newenvironment{shadingsec}{\colorlet{shadecolor}{chapcol!80}\\begin{shaded*} }{\end{shaded*}  }\n' \
                                                                                               '\\newenvironment{shadingsubsec}{\colorlet{shadecolor}{chapcol!60}\\begin{shaded*} }{\end{shaded*}  }\n' \
                                                                                               '\\newenvironment{shadingsubsubsec}{\colorlet{shadecolor}{chapcol!0}\\begin{shaded*} }{\end{shaded*}  }\n' \
                                                                                               '\\newenvironment{shadingsubsubsubsec}{\colorlet{shadecolor}{chapcol!0}\\begin{shaded*} }{\end{shaded*}  }\n' \
                                                                                               '\\newenvironment{shadingsubsubsubsubsec}{\colorlet{shadecolor}{chapcol!0}\\begin{shaded*} }{\end{shaded*}  }\n' \
                                                                                               ' \n' \
                                                                                               '\definecolor{chapcol}{rgb}{0.05 0.3 0.6}\n' \
                                                                                               ' \n' \
                                                                                               '\\titleformat{\section}[display]{\\normalfont\color{white!100!chapcol}\\normalfont}{}{0em}{\\begin {shadingsec}\\thesection\; #1\end{shadingsec}}\n' \
                                                                                               '\\titleformat{\subsection}[display]{\small\color{white!100!chapcol}\small}{}{0em}{\\begin {shadingsubsec}\\thesubsection\; #1\end{shadingsubsec}}\n' \
                                                                                               '\\titleformat{\subsubsubsection}[display]{\small\color{white!0!chapcol}\small}{}{0em}{\\begin {shadingsubsubsec}\\thesubsubsection\; #1\end{shadingsubsubsec}}\n' \
                                                                                               '\\titleformat{\subsubsubsection}[display]{\small\color{white!0!chapcol}\small}{}{0em}{\\begin {shadingsubsubsubsec}\\thesubsubsubsection\; #1\end{shadingsubsubsubsec}}\n' \
                                                                                               '\\titleformat{\subsubsubsubsection}[display]{\small\color{white!0!chapcol}\small}{}{0em}{\\begin {shadingsubsubsubsubsec}\\thesubsubsubsubsection\; #1\end{shadingsubsubsubsubsec}}\n' \
                                                                                               ' \n' \
                                                                                               '\setlength{\\tabcolsep}{2pt}\n' \
                                                                                               ' \n' \
                                                                                               '\\begin{document}\n' \
                                                                                               ' \n' \
                                                                                               'StandardParseBODYKEY\n' \
                                                                                               ' \n' \
                                                                                               '\end{document}\n'

        retText = retText.replace('$KEY_DOCNAME', o.docName)

        return retText

    def addColor(o, key, rgbColorVector):
        """
        adding a color to the LaTex replacement list. A color can be added
        like AddColor('$myKey', [1 0 0]) which would add red usable
        everywhere in the text input as $myKey{ TEXT OR NUMBER GOES HERE }.
        In the same way the default colors are usable.

        :param key:
        :param rgbColorVector:
        :return:
        """
        myNewColor = str(rgbColorVector).replace("[", "").replace("]", "")
        valueSet = '\color[rgb]{' + myNewColor + '}'
        o.ColorMap[key] = valueSet

    def addAlign(o, inMathFormula, key='StandardParseBODYKEY'):
        """
        simply adding the latex alignment environment

        :param inMathFormula:
        :param key:
        :return:
        """
        locText = '\\begin{align} ' + inMathFormula + ' \end{align}'
        o.addText(locText, key=key)

    def addLaTeXFigure(o, inLaTeXPlot, key='StandardParseBODYKEY', caption='', figureType='figure', where='h'):
        """
        Takes the Latex String of the form '\\addplot[color = Yellow]
        coordinates { (0,23.1)(10,27.5) };...' and adds it as text to the Tex
        file.
        where = 'h', 'H', 't', 'b', see LaTeX documentation.
        figureType = 'figure', 'sidewaysfigure'

        :param inLaTeXPlot:
        :param key:
        :param caption:
        :param figureType:
        :param where:
        :return:
        """

        locFigureString = '\\begin{figure}[' + where + ']\n' \
                                                       '\\begin{center}\n' \
                                                       'KEYtikzpicture\n' \
                                                       '\\end{center}\n' \
                                                       '\\caption{KEYcaption}\n' \
                                                       '\\end{figure}\n'
        if figureType.lower() == 'sidewaysfigure':
            locFigureString = ['\\begin{landscape}\n' + locFigureString + '\end{landscape}\n']

        # TODO: allow a list of plots to be added in one line:
        # if iscell(inLaTeXPlot):
        #    for i=1:length(inLaTeXPlot)
        #        inLaTeXPlot{i}.width = 1/length(inLaTeXPlot) - 0.01
        #        locFigureString = strrep(locFigureString,'KEYtikzpicture',[inLaTeXPlot{i}.GetLaTeXPlot() '\nKEYtikzpicture' ])
        #    end
        # else

        inLaTeXPlot.width = 1 - 0.01
        locFigureString = locFigureString.replace('KEYtikzpicture', inLaTeXPlot.getLaTeXPlot() + '\nKEYtikzpicture')

        locFigureString = locFigureString.replace('KEYcaption', caption)
        locFigureString = locFigureString.replace('figureTypeKEY', figureType)
        locFigureString = locFigureString.replace('KEYtikzpicture', '')
        o.addText(locFigureString, key=key)

    def compile(o):
        """
        This method combines the input which is given at this stage and tries
        to write the LaTeX file and tries to compile it. Error output is
        displayed in the command window.
        TODO: add proper try catch to close file.

        :return:
        """
        o.loadDefault()

        locKeyList = list(o.keyValueReplacementList.keys())
        locColorKeyList = list(o.ColorMap.keys())

        print('deleting...')
        try:
            os.remove(o.locURL)
        except OSError:
            pass

        print('parsing...')
        fout = open(o.locURL, mode='w')
        if fout == -1:
            print('could not find path:  ' + o.locURL)

        s = o.templateString
        # inserting the content parts:
        for key in o.usedKeys:
            s = s.replace(key, o.keyValueReplacementList[key])

        # inserting the color parts:
        for i in range(0, len(locColorKeyList)):
            col = o.ColorMap[locColorKeyList[i]]
            s = s.replace('KEYDefinedColors', '\definecolor{' + locColorKeyList[i][2:] + '}{rgb}{' + (
            col[13: - 2]) + '}\n' + 'KEYDefinedColors')  # e.g., \definecolor{Blue}{rgb}{0.05 0.3 0.6}
            s = s.replace(locColorKeyList[i] + '{',
                          '{' + o.ColorMap[locColorKeyList[i]])  # shifting the "{" for latex purposes.

        s = s.replace('KEYDefinedColors', '')

        print('writing to file...')
        fout.write(s)
        fout.close()

        print('LaTeX compiling pdflatex...')
        command = 'cd ' + o.working_directory + ' && lualatex -shell-escape  ' + o.destinationFileName + '.tex  --aux-directory=.\\auxiliaries-global '
        # [status, cmdout] = system(command, '-echo');
        # [status, cmdout] = system(command, '-echo'); % 2 times for updating the TOC

        # p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr = subprocess.PIPE)
        os.system(command)
        os.system(command)  # compile twice to update TOC and citations,etc.

        # out, err = p.communicate()
        # print(out)

    def loadDefault(o):
        """
        Determine the default text in which the relevant parts will be parsed.

        :return:
        """

        print('determening template structure...')
        o.templateString = ''
        if o.templateFileName != '':  # then use template file
            fin = open(o.working_directory + o.templateFileName, mode='r', encoding='utf-8')
            o.templateString = fin.read()
            fin.close()
        else:  # use pre defined template file TODO: add switch for more Layout options here
            o.templateString = o.standardLayout()
