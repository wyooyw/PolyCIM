import sys
class Tee:
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        
    def __enter__(self):
        class TeeOutput:
            def __init__(self, file, original_stream):
                self.file = file
                self.original_stream = original_stream
            
            def write(self, data):
                self.file.write(data)
                self.original_stream.write(data)
                
            def flush(self):
                self.file.flush()
                self.original_stream.flush()
        
        self.stdout_tee = TeeOutput(self.file, self.stdout)
        self.stderr_tee = TeeOutput(self.file, self.stderr)
        sys.stdout = self.stdout_tee
        sys.stderr = self.stderr_tee
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()