## **1. 개요**

- **프로젝트 명칭**: CleanZone
- **목적**: 피부에 대해 관심을 많이 갖고, 잘못된 식습관이나 스트레스로 인해 피부에 대한 고민을 가진 사람들이 많지만 병원을 갈 시간은 부족하고, 어떠한 제품을 써야하는지 고민하는 사람들이 많아지고 있습니다. 그럴 때 이 웹은 의사선생님과 화상상담으로 병원까지 갈 시간을 단축할 수 있고,
자신이 찍은 사진으로 의사선생님과 상담을 하며, 자신의 피부관리상태를 확인할 수 있고, 시술을 받을 때 피부과 예약을 잡아 줄 수 있는 서비스를 제공하려는 기획을 하였습니다.
- **대상 사용자**: 피부 고민이 있어 간단하게 진단을 받아 상담을 받고 싶은 사용자
- **기대 효과**: 피부과를 직접 방문하지 않아도, 간단하게 피부 진단을 받으며 병원 예약을 할 수 있는 서비스

## **2. 기능 요구 사항**
    
### **2.1 사용자 인터페이스**

- **홈페이지**: 회원 로그인과 AI피부진단, AI 챳봇과 병원 채팅 문의 시스템
- **로그인/회원가입 페이지**: jwt 및 개인정보 인증을 통한 사용자 인증 시스템
- **프로필 페이지**: 자신이 받았던 피부 진단에 대한 결과를 모아볼 수 있는 시스템
- **AI ChatBot**: 사용자가 피부에 관한 질문을 가볍게 물어 볼 수 있는 서비스 시스템

### **2.2 백엔드 서비스**

- **데이터베이스 관리**: 사용자 정보, 병원 정보, 사용자의 피부 진단 결과
- **RESTful API**: 프론트엔드와 백엔드, AI Server 통신할 수 있는 API 제공
- **보안**: 사용자 데이터 보호를 위한 암호화 및 보안 조치

