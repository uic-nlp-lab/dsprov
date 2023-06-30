-- make a word document read-only but allow comments
tell application "Microsoft Word"
   set source to POSIX file "{doc_path}"
   set doc to open file name source
   log "opened doc: " & name of doc
   protect doc protection type allow only comments
   save doc
   close doc
end tell
