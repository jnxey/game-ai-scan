[Setup]
AppName=AiScan
AppVersion=1.0.0
DefaultDirName={pf}\AiScan
DefaultGroupName=AiScan
OutputDir=output
OutputBaseFilename=AiScan_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "AiScan.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
; 开始菜单快捷方式
Name: "{group}\AiScan"; Filename: "{app}\AiScan.exe"; IconFilename: "{app}\AiScan.ico"
; 桌面快捷方式
Name: "{commondesktop}\AiScan"; Filename: "{app}\AiScan.exe"; IconFilename: "{app}\AiScan.ico"

[Registry]
; 设置开机自启动（当前用户）
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
ValueType: string; ValueName: "AiScan"; ValueData: """{app}\AiScan.exe"""; Flags: uninsdeletevalue

