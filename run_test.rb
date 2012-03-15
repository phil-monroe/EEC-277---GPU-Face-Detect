#!/usr/bin/env ruby
system "clear"
system "make"
system "clear"

puts "Running tests"

face_files      = `ls test/faces`.split
non_face_files  = `ls test/non-faces`.split
glasses_files   = `ls test/glasses`.split

puts "Number of faces:      #{face_files.count}"
puts "Number of non-faces:  #{non_face_files.count}"
puts "Number of glasses:    #{glasses_files.count}"

puts "Testing face files"
found_faces = 0
found_glasses = 0
for file in face_files
  output = `./detect test/faces/#{file} 2>&1`
  if output.index("A face was detected")
    puts "Face Detected in #{file}"
    found_faces += 1
  end
  if output.index("Glasses were detected")
    puts "Glasses were detected in #{file}"
    found_glasses += 1
  end
  if output.index("error")
    puts "Failed on file #{file}"
    exit -1
  end
end
puts "Found #{found_faces} faces in #{face_files.count} files."
puts "Found #{found_glasses} glasses in #{glasses_files.count} files."
puts


puts "Testing non-face files"
found_faces = 0
found_glasses = 0
for file in non_face_files
  output = `./detect test/non-faces/#{file} 2>&1`
  if output.index("A face was detected")
    puts "Face Detected in #{file}"
    found_faces += 1
  end
  if output.index("Glasses were detected")
    puts "Glasses were detected in #{file}"
    found_glasses += 1
  end
  if output.index("error")
    puts "Failed on file #{file}"
    exit -1
  end
end
puts "Found #{found_faces} faces in #{non_face_files.count} files."
puts "Found #{found_glasses} glasses in #{glasses_files.count} files."
puts


puts "Testing glasses files"
found_faces = 0
found_glasses = 0
for file in glasses_files
  output = `./detect test/glasses/#{file} 2>&1`
  if output.index("A face was detected")
    puts "Face Detected in #{file}"
    found_faces += 1
  end
  if output.index("Glasses were detected")
    puts "Glasses were detected in #{file}"
    found_glasses += 1
  end
  if output.index("error")
    puts "Failed on file #{file}"
    exit -1
  end
end
puts "Found #{found_faces} faces in #{glasses_files.count} files."
puts "Found #{found_glasses} glasses in #{glasses_files.count} files."

