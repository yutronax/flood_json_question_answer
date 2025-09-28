import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5Model:
  def __init__(self):
      self.model_path = "/contet/best_model"

  def load_trained_model(self):
     
      print(f"Model yükleniyor: {self.model_path}")

      tokenizer = T5Tokenizer.from_pretrained(self.model_path)
      model = T5ForConditionalGeneration.from_pretrained(self.model_path)

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      model.to(device)
      model.eval()

      print(f"Model yüklendi - Device: {device}")
      return model, tokenizer, device

  def test_model(self,question, objects, model=None, tokenizer=None, device=None,):


      # Model yüklenmemişse yükle
      if model is None:
          model, tokenizer, device = self.load_trained_model()

      # Context oluştur (orijinal format)
      context_parts = []
      for obj, count in objects.items():
          context_parts.append(f"{obj.replace('_', ' ')} ({count})")
      context = "Objects present: " + ", ".join(context_parts) + "."

      # Input text (training format ile aynı)
      input_text = f"Context: {context} Question: {question}"



      # Tokenize
      inputs = tokenizer(
          input_text,
          return_tensors='pt',
          max_length=256,
          truncation=True,
          padding=True
      ).to(device)

      # Generate answer
      with torch.no_grad():
          outputs = model.generate(
              **inputs,
              max_length=64,
              num_beams=4,
              early_stopping=True,
              do_sample=False,    # Deterministik sonuç
              temperature=1.0
          )


      answer = tokenizer.decode(outputs[0], skip_special_tokens=True)



      return answer

  def batch_test(self,test_cases):
     

      # Model'i bir kere yükle
      model, tokenizer, device = self.load_trained_model()



      results = []

      for i, (question, objects, expected) in enumerate(test_cases):
          print(f"\n--- Test {i+1}/{len(test_cases)} ---")

          # Test et
          answer = self.test_model(question, objects, model, tokenizer, device)

          # Sonucu kaydet
          result = {
              'question': question,
              'objects': objects,
              'expected': expected,
              'predicted': answer,
              'correct': str(answer).strip() == str(expected).strip()
          }
          results.append(result)

         
         
      # Özet
      correct_count = sum(1 for r in results if r['correct'])
      accuracy = correct_count / len(results) * 100

    

      return results
