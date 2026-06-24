#ifndef GUI_H
#define GUI_H

#include <wx/wx.h>
#include <wx/dir.h>
#include <wx/filename.h>
#include <wx/textctrl.h>
#include <wx/spinctrl.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/stattext.h>
#include <wx/filepicker.h>
#include <wx/thread.h>

#include <thread>
#include <mutex>
#include <vector>
#include <string>

class StereoApp : public wxApp
{
public:
    virtual bool OnInit() wxOVERRIDE;
};

class StereoFrame : public wxFrame
{
public:
    StereoFrame();
    void OnRun(wxCommandEvent& event);
    void OnStop(wxCommandEvent& event);
    void OnClassify(wxCommandEvent& event);

private:
    void RunReconstruction();          // runs in worker thread
    void RunClassification();          // runs in worker thread
    void OnReconstructionFinished();   // called on main thread
    void OnClassificationFinished();   // called on main thread

    wxDirPickerCtrl* m_folderPicker;      // stereo folder
    wxDirPickerCtrl* m_bgFolderPicker;    // background folder
    wxFilePickerCtrl* m_txtPicker;
    wxFilePickerCtrl* m_camPicker;
    wxSpinCtrl* m_stepSpin;
    wxTextCtrl* m_outputText;
    wxButton* m_runButton;
    wxButton* m_stopButton;
    wxButton* m_classifyButton;

    std::thread m_workerThread;
    bool m_running = false;
    bool m_classifying = false;
    std::mutex m_mutex;
};

#endif // GUI_H