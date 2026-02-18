#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;

const INPUT_DIR: &str = "A:/omni read/input";

#[tauri::command]
fn save_uploaded_image(filename: String, data: Vec<u8>) -> Result<String, String> {
    let input_dir = PathBuf::from(INPUT_DIR);
    fs::create_dir_all(&input_dir).map_err(|e| e.to_string())?;

    let safe_filename = PathBuf::from(filename)
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or_else(|| "Nom de fichier invalide".to_string())?
        .to_string();

    let dest = input_dir.join(safe_filename);
    fs::write(&dest, data).map_err(|e| e.to_string())?;

    Ok(dest.to_string_lossy().replace('\\', "/"))
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![save_uploaded_image])
        .run(tauri::generate_context!())
        .expect("error while running tauri app");
}
