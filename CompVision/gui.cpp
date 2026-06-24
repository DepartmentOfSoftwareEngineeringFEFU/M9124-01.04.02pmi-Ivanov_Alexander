#include "gui.h"
#include "object_search.h"
#include "combined_view.h"

#include <iostream>
#include <streambuf>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <algorithm>

class GUILogStreambuf : public std::streambuf
{
public:
    GUILogStreambuf(wxTextCtrl* textCtrl)
        : m_textCtrl(textCtrl)
    {
        setp(m_buffer, m_buffer + BUFFER_SIZE - 1);
    }

protected:
    virtual int overflow(int c) override
    {
        if (c != EOF)
        {
            *pptr() = static_cast<char>(c);
            pbump(1);
        }
        return sync() == 0 ? c : EOF;
    }

    virtual int sync() override
    {
        if (pbase() != pptr())
        {
            std::string line(pbase(), pptr() - pbase());
            wxTheApp->CallAfter([this, line]() {
                if (m_textCtrl)
                    m_textCtrl->AppendText(wxString::FromUTF8(line));
                });
            setp(m_buffer, m_buffer + BUFFER_SIZE - 1);
        }
        return 0;
    }

private:
    wxTextCtrl* m_textCtrl;
    static constexpr size_t BUFFER_SIZE = 4096;
    char m_buffer[BUFFER_SIZE];
};

bool StereoApp::OnInit()
{
    StereoFrame* frame = new StereoFrame();
    frame->Show(true);
    return true;
}

StereoFrame::StereoFrame()
    : wxFrame(nullptr, wxID_ANY, "Стерео реконструкция", wxDefaultPosition, wxSize(700, 600))
{
    wxStaticText* bgLabel = new wxStaticText(this, wxID_ANY, "Папка с фоновыми изображениями:");
    m_bgFolderPicker = new wxDirPickerCtrl(this, wxID_ANY, wxEmptyString,
        "Выберите папку с фоновыми изображениями",
        wxDefaultPosition, wxDefaultSize,
        wxDIRP_DEFAULT_STYLE | wxDIRP_USE_TEXTCTRL);

    wxStaticText* folderLabel = new wxStaticText(this, wxID_ANY, "Папка со стерео-снимками:");
    m_folderPicker = new wxDirPickerCtrl(this, wxID_ANY, wxEmptyString,
        "Выберите папку со стерео-снимками",
        wxDefaultPosition, wxDefaultSize,
        wxDIRP_DEFAULT_STYLE | wxDIRP_USE_TEXTCTRL);

    wxStaticText* txtLabel = new wxStaticText(this, wxID_ANY, "Файл с номерами снимков:");
    m_txtPicker = new wxFilePickerCtrl(this, wxID_ANY, wxEmptyString,
        "Выберите .txt файл",
        "*.txt", wxDefaultPosition, wxDefaultSize,
        wxFLP_DEFAULT_STYLE | wxFLP_USE_TEXTCTRL);

    wxStaticText* camLabel = new wxStaticText(this, wxID_ANY, "Файл параметров камеры:");
    m_camPicker = new wxFilePickerCtrl(this, wxID_ANY, wxEmptyString,
        "Выберите файл параметров камеры",
        "*.txt", wxDefaultPosition, wxDefaultSize,
        wxFLP_DEFAULT_STYLE | wxFLP_USE_TEXTCTRL);

    m_classifyButton = new wxButton(this, wxID_ANY, "Найти снимки с объектами");
    m_runButton = new wxButton(this, wxID_ANY, "Начать реконструкцию");
    m_stopButton = new wxButton(this, wxID_ANY, "Остановить");
    m_stopButton->Enable(false);

    m_outputText = new wxTextCtrl(this, wxID_ANY, wxEmptyString,
        wxDefaultPosition, wxDefaultSize,
        wxTE_MULTILINE | wxTE_READONLY | wxTE_RICH);
    wxFont cyrillicFont(wxFontInfo(10).FaceName("Arial"));
    m_outputText->SetFont(cyrillicFont);

    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
    wxFlexGridSizer* grid = new wxFlexGridSizer(2, 5, 5);
    grid->AddGrowableCol(1, 1);

    grid->Add(bgLabel, 0, wxALIGN_CENTER_VERTICAL);
    grid->Add(m_bgFolderPicker, 1, wxEXPAND);
    grid->Add(folderLabel, 0, wxALIGN_CENTER_VERTICAL);
    grid->Add(m_folderPicker, 1, wxEXPAND);
    grid->Add(txtLabel, 0, wxALIGN_CENTER_VERTICAL);
    grid->Add(m_txtPicker, 1, wxEXPAND);
    grid->Add(camLabel, 0, wxALIGN_CENTER_VERTICAL);
    grid->Add(m_camPicker, 1, wxEXPAND);

    mainSizer->Add(grid, 0, wxEXPAND | wxALL, 10);

    wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
    buttonSizer->Add(m_classifyButton, 0, wxRIGHT, 10);
    buttonSizer->Add(m_runButton, 0, wxRIGHT, 10);
    buttonSizer->Add(m_stopButton, 0, wxLEFT, 10);
    mainSizer->Add(buttonSizer, 0, wxALIGN_CENTER | wxALL, 10);

    mainSizer->Add(m_outputText, 1, wxEXPAND | wxALL, 10);
    SetSizer(mainSizer);

    m_runButton->Bind(wxEVT_BUTTON, &StereoFrame::OnRun, this);
    m_stopButton->Bind(wxEVT_BUTTON, &StereoFrame::OnStop, this);
    m_classifyButton->Bind(wxEVT_BUTTON, &StereoFrame::OnClassify, this);
}

void StereoFrame::OnClassify(wxCommandEvent& WXUNUSED(event))
{
    if (m_running || m_classifying) return;

    wxString stereoFolder = m_folderPicker->GetPath();
    wxString bgFolder = m_bgFolderPicker->GetPath();

    if (stereoFolder.empty() || !wxDirExists(stereoFolder)) {
        m_outputText->AppendText("Ошибка: выберите корректную папку со стерео снимками.\n");
        return;
    }
    if (bgFolder.empty() || !wxDirExists(bgFolder)) {
        m_outputText->AppendText("Ошибка: выберите корректную папку с фоновыми изображениями.\n");
        return;
    }

    m_outputText->Clear();
    m_outputText->AppendText("Запуск классификации...\n");

    m_classifying = true;
    m_classifyButton->Enable(false);
    m_runButton->Enable(false);

    m_workerThread = std::thread([this, stereoFolder, bgFolder]() {
        GUILogStreambuf logBuf(m_outputText);
        std::streambuf* oldCout = std::cout.rdbuf(&logBuf);

        try {
            std::string outputFile = (stereoFolder + "\\object_frames.txt").ToStdString();
            classifyStereoImages(bgFolder.ToStdString(),
                stereoFolder.ToStdString(),
                outputFile,
                1.0,      // thresh_k
                99.0);    // percentile
            std::cout << "Классификация завершена. Результат сохранён в: " << outputFile << "\n";
        }
        catch (const std::exception& e) {
            std::cout << "Исключение в классификации: " << e.what() << "\n";
        }
        catch (...) {
            std::cout << "Неизвестное исключение в классификации.\n";
        }

        std::cout.rdbuf(oldCout);
        wxTheApp->CallAfter([this]() { OnClassificationFinished(); });
        });
}

void StereoFrame::OnClassificationFinished()
{
    if (m_workerThread.joinable())
        m_workerThread.join();

    m_classifying = false;
    m_classifyButton->Enable(true);
    m_runButton->Enable(true);
    m_outputText->AppendText("\n--- Поиск объектов завершён ---\n");
}

void StereoFrame::OnRun(wxCommandEvent& WXUNUSED(event))
{
    if (m_running || m_classifying) return;

    m_outputText->Clear();

    wxString folder = m_folderPicker->GetPath();
    wxString txtFile = m_txtPicker->GetPath();
    wxString camFile = m_camPicker->GetPath();

    if (folder.empty() || !wxDirExists(folder)) {
        m_outputText->AppendText("Ошибка: выберите корректную папку.\n");
        return;
    }
    if (txtFile.empty() || !wxFileExists(txtFile)) {
        m_outputText->AppendText("Ошибка: выберите корректный .txt файл с номерами снимков.\n");
        return;
    }
    if (camFile.empty() || !wxFileExists(camFile)) {
        m_outputText->AppendText("Ошибка: выберите корректный файл параметров камеры.\n");
        return;
    }

    wxArrayString allFiles;
    wxDir::GetAllFiles(folder, &allFiles, wxEmptyString, wxDIR_FILES);

    std::map<int, bool> hasLeft, hasRight;
    for (const auto& f : allFiles) {
        wxString fname = wxFileName(f).GetFullName();
        wxArrayString parts = wxSplit(fname, '_');
        if (parts.GetCount() != 3) continue;
        if (parts[0] != "frame") continue;

        wxString numStr = parts[1];
        long num;
        if (!numStr.ToLong(&num)) continue;

        wxString sidePart = parts[2];
        if (!sidePart.EndsWith(".bmp")) continue;
        wxString sideStr = sidePart.BeforeLast('.');
        if (sideStr != "0" && sideStr != "1") continue;

        int number = (int)num;
        if (sideStr == "0")
            hasLeft[number] = true;
        else
            hasRight[number] = true;
    }

    std::set<int> pairedNumbers;
    for (const auto& pair : hasLeft) {
        int n = pair.first;
        if (hasRight.find(n) != hasRight.end())
            pairedNumbers.insert(n);
    }

    if (pairedNumbers.empty()) {
        m_outputText->AppendText("Ошибка: стереопары не найдены. Ожидаются файлы 'frame_DDDDD_0.bmp' и 'frame_DDDDD_1.bmp'.\n");
        return;
    }

    int maxNum = *pairedNumbers.rbegin();
    for (int i = 0; i <= maxNum; ++i) {
        if (pairedNumbers.find(i) == pairedNumbers.end()) {
            m_outputText->AppendText("Ошибка: номера кадров не образуют непрерывный ряд от 0 до " +
                wxString::Format("%d", maxNum) + ".\n");
            return;
        }
    }
    int totalFrames = maxNum + 1;
    m_outputText->AppendText(wxString::Format("Найдено %d кадров (0..%d).\n", totalFrames, maxNum));

    std::vector<int> frameSequence;
    std::ifstream infile(txtFile.ToStdString());
    if (!infile.is_open()) {
        m_outputText->AppendText("Ошибка: не удалось открыть текстовый файл.\n");
        return;
    }
    std::string line;
    while (std::getline(infile, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty()) continue;
        try {
            frameSequence.push_back(std::stoi(line));
        }
        catch (...) {
            m_outputText->AppendText(wxString::Format("Предупреждение: игнорируем нечисловую строку: '%s'\n", line));
        }
    }
    infile.close();

    if (frameSequence.empty()) {
        m_outputText->AppendText("Ошибка: не найдены корректные номера кадров в файле.\n");
        return;
    }

    for (int f : frameSequence) {
        if (f < 0 || f >= totalFrames) {
            m_outputText->AppendText(wxString::Format("Ошибка: кадр %d вне диапазона (0..%d).\n", f, totalFrames - 1));
            return;
        }
    }
    m_outputText->AppendText(wxString::Format("Прочитано %zu номеров кадров.\n", frameSequence.size()));

    m_running = true;
    m_runButton->Enable(false);
    m_stopButton->Enable(true);
    m_classifyButton->Enable(false);

    m_workerThread = std::thread([this, frameSequence, totalFrames, folder, camFile]() {
        GUILogStreambuf logBuf(m_outputText);
        std::streambuf* oldCout = std::cout.rdbuf(&logBuf);

        try {
            reconstructFromFrameSequence(frameSequence, totalFrames, 10, folder.ToStdString(), camFile.ToStdString());
        }
        catch (const std::exception& e) {
            std::cout << "Исключение в реконструкции: " << e.what() << "\n";
        }
        catch (...) {
            std::cout << "Неизвестное исключение в реконструкции.\n";
        }

        std::cout.rdbuf(oldCout);
        wxTheApp->CallAfter([this]() { OnReconstructionFinished(); });
        });
}

void StereoFrame::OnStop(wxCommandEvent& WXUNUSED(event))
{
    if (m_workerThread.joinable()) {
        m_workerThread.detach();
    }
    m_running = false;
    m_classifying = false;
    m_runButton->Enable(true);
    m_stopButton->Enable(false);
    m_classifyButton->Enable(true);
    m_outputText->AppendText("\n--- Остановлено пользователем (поток отсоединён) ---\n");
}

void StereoFrame::OnReconstructionFinished()
{
    if (m_workerThread.joinable())
        m_workerThread.join();

    m_running = false;
    m_runButton->Enable(true);
    m_stopButton->Enable(false);
    m_classifyButton->Enable(true);
    m_outputText->AppendText("\n--- Реконструкция завершена ---\n");
}